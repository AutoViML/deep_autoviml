############################################################################################
#Copyright 2021 Google LLC

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
############################################################################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
import tempfile
import pdb
import copy
import warnings
warnings.filterwarnings(action='ignore')
import functools
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
############################################################################################
# TensorFlow ≥2.4 is required
import tensorflow as tf
import tensorflow_text as text

np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, CategoryCrossing
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, Discretization, Hashing
from tensorflow.keras.layers import Embedding, Reshape, Dropout, Dense

from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

############################################################################################
# data pipelines
from deep_autoviml.data_load.classify_features import classify_features_using_pandas

from deep_autoviml.data_load.extract import fill_missing_values_for_TF2
from deep_autoviml.utilities.utilities import print_one_row_from_tf_dataset, print_one_row_from_tf_label
############################################################################################
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from IPython.core.display import Image, display
import pickle
##### Suppress all TF2 and TF1.x warnings ###################
try:
    tf.logging.set_verbosity(tf.logging.ERROR)
except:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
############################################################################################
from tensorflow.keras.layers import Reshape, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
#########################################################################################
import os
import pickle
import time
############################################################################################
def load_test_data(test_data_or_file, project_name, cat_vocab_dict="",
                                                 verbose=0):
    """
    Load a CSV file and given a project name, it will load the artifacts in project_name folder.
    Optionally you can provide the artifacts dictionary as "cat_vocab_dict" in this input.

    Outputs:
    --------
    data_batches: a tf.data.Dataset which will be Repeat batched dataset
    cat_vocab_dict: artifacts dictionary that you can feed to the predict function of model.
    """
    ### load a small sample of data into a pandas dataframe ##
    if isinstance(test_data_or_file, str):
        test_small = pd.read_csv(test_data_or_file) ### this reads the entire file
    else:
        test_small = copy.deepcopy(test_data_or_file)
    filesize = test_small.shape[0]
    print('Loaded test data size: %d' %filesize)
    #### All column names in Tensorflow should have no spaces ! So you must convert them here!
    sel_preds = ["_".join(x.split(" ")) for x in list(test_small) ]
    sel_preds = ["_".join(x.split("(")) for x in sel_preds ]
    sel_preds = ["_".join(x.split(")")) for x in sel_preds ]
    sel_preds = ["_".join(x.split("/")) for x in sel_preds ]
    sel_preds = ["_".join(x.split("\\")) for x in sel_preds ]
    sel_preds = [x.lower() for x in sel_preds ]

    test_small.columns = sel_preds

    print('Alert! Modified column names to satisfy rules for column names in Tensorflow...')

    ################### if cat_vocab_dict is not given, load it ####
    no_cat_vocab_dict = False
    if not cat_vocab_dict:
        ### You must load it from disk ###
        try:
            pickle_path = os.path.join(project_name, "cat_vocab_dict.pickle")
            print('\nLoading cat_vocab_dict file using pickle in %s...' %pickle_path)
            cat_vocab_dict = pickle.load(open(pickle_path,"rb"))
            print('    Loaded pickle file in %s' %pickle_path)
        except:
            print('Unable to load pickle file. Continuing...')
            no_cat_vocab_dict = True
    ####################################################
    if not no_cat_vocab_dict:
        target = cat_vocab_dict['target_variables']
        usecols = cat_vocab_dict['target_variables']
        if len(target) == 1:
            target_name = target[0]
        else:
            target_name = target
    else:
        target = []
        target_name = ''
        print('no target variable found since model artifacts dictionary could not be found')
    ### classify variables using the small dataframe ##
    model_options = {}

    if no_cat_vocab_dict:
        model_options['DS_LEN'] = 10000  ### Just set some default #######
        ###### Just send in entire dataframe to convert and correct dtypes using this function ##
        ######   If you don't do this, in some data sets due to mixed types it errors ###
        ######  Just send in target_name as '' since we want even target to be corrected if it
        #####    has the wrong data type since tensorflow automatically detects data types.
        test_small, var_df, cat_vocab_dict = classify_features_using_pandas(test_small, target='',
                                    model_options=model_options, verbose=verbose)
        ##########    Just transfer all the values from var_df to cat_vocab_dict  ##########
        for each_key in var_df:
            cat_vocab_dict[each_key] = var_df[each_key]
        ####################################################################################
    else:
        ###### Just send in entire dataframe to convert and correct dtypes using this function ##
        ######   If you don't do this, in some data sets due to mixed types it errors ###
        ######  Just send in target_name as '' since we want even target to be corrected if it
        #####    has the wrong data type since tensorflow automatically detects data types.
        model_options['DS_LEN'] = cat_vocab_dict['DS_LEN'] ### you need this to classify features
        test_small, _, _ = classify_features_using_pandas(test_small, target='',
                                    model_options=model_options, verbose=verbose)
    ############  Now load the file or dataframe into tf.data.DataSet here #################

    preds = list(test_small)
    #batch_size = 64   ## artificially set a size ###
    batch_size = cat_vocab_dict["batch_size"]
    cat_vocab_dict["DS_LEN"] = filesize
    num_epochs = None
    ###  Initially set this batch_size very high so that you can get the max(), min() and vocab to be realistic
    if isinstance(test_data_or_file, str):
        #### Set column defaults while reading dataset from CSV files - that way, missing values avoided!
        ### The following are valid CSV dtypes for missing values: float32, float64, int32, int64, or string
        ### fill all missing values in categorical variables with "None"
        ### Similarly. fill all missing values in float variables with -99
        if test_small.isnull().sum().sum() > 0:
            print('There are %d missing values in dataset - filling with default values...' %(
                                    test_small.isnull().sum().sum()))
        string_cols = test_small.select_dtypes(include='object').columns.tolist() + test_small.select_dtypes(
                                            include='category').columns.tolist()
        integer_cols =  test_small.select_dtypes(include='integer').columns.tolist()
        float_cols = test_small.select_dtypes(include='float').columns.tolist()
        column_defaults = [-99.0 if x in float_cols else -99 if x in integer_cols else "missing" for x in test_small]
        #### Once the missing data is filled, it's ready to load into tf.data.DataSet ############
        data_batches = tf.data.experimental.make_csv_dataset(test_data_or_file,
                                               batch_size=batch_size,
                                               column_names=preds,
                                               label_name=None,
                                               num_epochs = num_epochs,
                                               column_defaults=column_defaults,
                                               shuffle=False,
                                               num_parallel_reads=tf.data.experimental.AUTOTUNE)
    else:
        #### This is to load dataframes into datasets ########################
        if test_small.isnull().sum().sum() > 0:
            test_small = fill_missing_values_for_TF2(test_small, cat_vocab_dict)

        drop_cols = cat_vocab_dict['columns_deleted']
        if len(drop_cols) > 0:
            print('    Dropping %s columns from dataset...' %drop_cols)
            try:
                test_small.drop(drop_cols, axis=1, inplace=True)
                #### In some datasets, due to mixed data types in test_small, this next line errors. Beware!!
            except:
                print('    in some datasets, due to mixed data types in test, this errors. Continuing...')
        data_batches = tf.data.Dataset.from_tensor_slices(dict(test_small))
        ### batch it if you are creating it from a dataframe
        data_batches = data_batches.batch(batch_size, drop_remainder=False).repeat()

    print('    test data loaded successfully.')

    if verbose >= 1:
        try:
            print_one_row_from_tf_dataset(data_batches)
        except:
            pass
    #### These are the input variables for which we are going to create keras.Inputs ###\
    return data_batches, cat_vocab_dict
#################################################################################################
def lenopenreadlines(filename):
    with open(filename) as f:
        return len(f.readlines())
#################################################################################################
def find_batch_size(DS_LEN):
    ### Since you cannot deal with a very large dataset in pandas, let's look into how big the file is
    maxrows = 10000
    if DS_LEN < 100:
        batch_ratio = 0.16
    elif DS_LEN >= 100 and DS_LEN < 1000:
        batch_ratio = 0.05
    elif DS_LEN >= 1000 and DS_LEN < 10000:
        batch_ratio = 0.01
    elif DS_LEN >= maxrows and DS_LEN <= 100000:
        batch_ratio = 0.001
    else:
        batch_ration = 0.0001
    batch_len = int(batch_ratio*DS_LEN)
    print('    Batch size selected as %d' %batch_len)
    return batch_len
###############################################################################################
class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)
#####################################################################################
def load_model_dict(model_or_model_path, cat_vocab_dict, project_name, keras_model_type):
    start_time = time.time()
    if not cat_vocab_dict:
        ### No cat_vocab_dict is given. Hence you must load it from disk ###
        print('\nNo model artifacts file given. Loading cat_vocab_dict file using pickle. Will take time...')
        if isinstance(model_or_model_path, str):
            if model_or_model_path:
                try:
                    pickle_path = os.path.join(model_or_model_path,os.path.join("artifacts", "cat_vocab_dict.pickle"))
                    cat_vocab_dict = pickle.load(open(pickle_path,"rb"))
                    print('    Loaded pickle file in %s' %pickle_path)
                except:
                    print('Unable to load model and data artifacts cat_vocab_dictionary file. Returning...')
                    return []
            modeltype = cat_vocab_dict['modeltype']
        else:
            try:
                ### Since model_path is not given, we will use project_name and keras_model_type to find it ##
                pickle_path = os.path.join(project_name, keras_model_type)
                list_folders = os.listdir(pickle_path)
                folder_path = list_folders[0]
                pickle_path = os.path.join(pickle_path, folder_path)
                pickle_path = os.path.join(pickle_path, "artifacts")
                print('Selecting first artifacts file in folder %s. Change model path if you want different.' %folder_path)
                pickle_path = os.path.join(pickle_path, "cat_vocab_dict.pickle")
                cat_vocab_dict = pickle.load(open(pickle_path,"rb"))
                print('    Loaded pickle file in %s' %pickle_path)
                modeltype = cat_vocab_dict['modeltype']
            except:
                print('Error: No path for model artifacts such as model_path given. Returning')
                return
    else:
        ### cat_vocab_dictionary is given #####
            modeltype = cat_vocab_dict['modeltype']
    ### Check if model is available to be loaded #######
    print('\nLoading deep_autoviml model from %s folder. This will take time...' %model_or_model_path)
    if isinstance(model_or_model_path, str):
        try:
            if model_or_model_path == "":
                model_or_model_path = os.path.join(project_name, keras_model_type)
            else:
                if modeltype == 'Regression':
                    model = tf.keras.models.load_model(os.path.join(model_or_model_path))
                else:
                    model = tf.keras.models.load_model(os.path.join(model_or_model_path),
                            custom_objects={'BalancedSparseCategoricalAccuracy': BalancedSparseCategoricalAccuracy})
        except Exception as error:
            print('Could not load given model.\nError: %s\n Please check your model path and try again.' %error)
            return
    else:
        print('\nUsing %s model provided as input...' %model_or_model_path)
        model = model_or_model_path
    print('Time taken to load saved model = %0.0f seconds' %((time.time()-start_time)))
    return model, cat_vocab_dict
###################################################################################################
##########     THIS IS THE MAIN PREDICT() FUNCTION            #####################################
###################################################################################################
def predict(model_or_model_path, project_name, test_dataset,
                    keras_model_type, cat_vocab_dict="", verbose=0):
    start_time2 = time.time()
    model, cat_vocab_dict = load_model_dict(model_or_model_path, cat_vocab_dict, project_name, keras_model_type)
    ##### load the test data set here #######
    if keras_model_type.lower() in ['nlp', 'text']:
        NLP_VARS = cat_vocab_dict['predictors_in_train']
    else:
        NLP_VARS = cat_vocab_dict['nlp_vars']
    ################################################################
    @tf.function
    def process_NLP_features(features):
        """
        This is how you combine all your string NLP features into a single new feature.
        Then you can perform embedding on this combined feature.
        It takes as input an ordered dict named features and returns the same features format.
        """
        return tf.strings.reduce_join([features[i] for i in NLP_VARS],axis=0,
                keepdims=False, separator=' ', name="combined")
    ################################################################
    NLP_COLUMN = "combined_nlp_text"
    ################################################################
    @tf.function
    def combine_nlp_text(features):
        ##use x to derive additional columns u want. Set the shape as well
        y = {}
        y.update(features)
        y[NLP_COLUMN] = tf.strings.reduce_join([features[i] for i in NLP_VARS],axis=0,
                keepdims=False, separator=' ')
        return y
    ################################################################
    if isinstance(test_dataset, str):
        test_ds, cat_vocab_dict2 = load_test_data(test_dataset, project_name=project_name,
                                cat_vocab_dict=cat_vocab_dict, verbose=verbose)
        ### You have to load only the NLP or text variables into dataset. otherwise, it will fail during predict
        batch_size = cat_vocab_dict2["batch_size"]
        if NLP_VARS:
            if keras_model_type.lower() in ['nlp', 'text']:
                test_ds = test_ds.map(process_NLP_features)
                test_ds = test_ds.unbatch().batch(batch_size)
                print('    combined NLP or text vars: %s into a single feature successfully' %NLP_VARS)
            else:
                test_ds = test_ds.map(combine_nlp_text)
                print('    combined NLP or text vars: %s into a single feature successfully' %NLP_VARS)
        else:
            print('No NLP vars in data set. No preprocessing done.')
        DS_LEN = cat_vocab_dict2["DS_LEN"]
        print("test data size = ",DS_LEN, ', batch_size = ',batch_size)
    elif isinstance(test_dataset, pd.DataFrame) or isinstance(test_dataset, pd.Series):
        if keras_model_type.lower() in ['nlp', 'text']:
            #### You must only load the text or nlp columns into the dataset. Otherwise, it will fail during predict.
            test_dataset = test_dataset[NLP_VARS]
        test_ds, cat_vocab_dict2 = load_test_data(test_dataset, project_name=project_name,
                                cat_vocab_dict=cat_vocab_dict, verbose=verbose)
        batch_size = cat_vocab_dict2["batch_size"]
        DS_LEN = cat_vocab_dict2["DS_LEN"]
        print("test data size = ",DS_LEN, ', batch_size = ',batch_size)
        if NLP_VARS:
            if keras_model_type.lower() in ['nlp', 'text']:
                test_ds = test_ds.map(process_NLP_features)
                test_ds = test_ds.unbatch().batch(batch_size)
                print('    processed NLP and text vars: %s successfully' %NLP_VARS)
            else:
                test_ds = test_ds.map(combine_nlp_text)
                print('    combined NLP or text vars: %s into a single combined_nlp_text successfully' %NLP_VARS)
        else:
            print('No NLP vars in data set. No preprocessing done.')
    else:
        ### It must be a tf.data.Dataset hence just load it as is ####
        if cat_vocab_dict:
            DS_LEN = cat_vocab_dict["DS_LEN"]
            batch_size = cat_vocab_dict["batch_size"]
        else:
            print('Since artifacts dictionary (cat_vocab_dict) not provided, using 100,000 as default test data size and batch size as 64.')
            print('    if you want to modify them, send in cat_vocab_dict["DS_LEN"] and cat_vocab_dict["batch_size"]')
            DS_LEN = 100000
            batch_size = 64
        test_ds = test_dataset
        if NLP_VARS:
            if keras_model_type.lower() in ['nlp', 'text']:
                test_ds = test_ds.map(process_NLP_features)
                test_ds = test_ds.unbatch().batch(batch_size)
                print('    processed NLP and text vars: %s successfully' %NLP_VARS)
            else:
                test_ds = test_ds.map(combine_nlp_text)
                print('    combined NLP or text vars: %s into a single combined_nlp_text successfully' %NLP_VARS)
        else:
            print('No NLP vars in data set. No preprocessing done.')
        cat_vocab_dict2 = copy.deepcopy(cat_vocab_dict)
    ##################################################################
    BOOLS = cat_vocab_dict2['bools']
    #################################################################################
    @tf.function
    def process_boolean_features(features):
        """
        This is how you convert all your boolean features into float variables.
        The reason you have to do this is because tf.keras does not know how to handle boolean types.
        It takes as input an ordered dict named features and returns the same in features format.
        """
        for feature_name in features:
            if feature_name in BOOLS:
                # Cast boolean feature values to string.
                features[feature_name] = tf.cast(features[feature_name], tf.dtypes.float32)
        return features
    ##################################################################
    try:
        test_ds = test_ds.map(process_boolean_features)
        print('Boolean column successfully processed')
    except:
        print('Error in Boolean column processing. Continuing...')
    ## num_steps is needed to predict on whole dataset once ##
    try:
        num_steps = int(np.ceil(DS_LEN/batch_size))
        print('Batch size = %s' %batch_size)
    except:
        num_steps = 1
    #########  See if you can predict here if not return the null result #####
    print('    number of steps needed to predict: %d' %num_steps)
    y_test_preds_list = []
    targets = cat_vocab_dict2['target_variables']
    ##### Now you need to save the predictions ###
    modeltype = cat_vocab_dict2['modeltype']
    num_labels = cat_vocab_dict2['num_labels']
    num_classes = cat_vocab_dict2['num_classes']    
    ####### save the predictions only upto input size ###
    ########  This is where we start predictions on test data set ##############
    try:
        y_probas = model.predict(test_ds, steps=num_steps)
    except:
        print('predictions from model erroring. Check your model and test data and retry again.')
        return
    ######  Now convert the model predictions into classes #########
    try:
        y_test_preds_list = convert_predictions_from_model(y_probas, cat_vocab_dict2, DS_LEN)
    except:
        print('Converting model predictions into classes or other forms is erroring. Convert it yourself.')
        return y_probas


    #####  We now show how many items are in the output  ###################
    print('Returning model predictions in form of a list...of length %d' %len(y_test_preds_list))
    print('Time taken in mins for predictions = %0.0f' %((time.time()-start_time2)/60))
    return y_test_preds_list
############################################################################################
def convert_predictions_from_model(y_probas, cat_vocab_dict, DS_LEN):
    y_test_preds_list = []
    target = cat_vocab_dict['target_variables']
    modeltype = cat_vocab_dict["modeltype"]
    if isinstance(target, list):
        if len(target) == 1:
            target = target[0]
    #### This is where predictions are converted back to classes ###
    if isinstance(target, str):
        if modeltype != 'Regression':
            #### This is for Single Label classification problems ######
            y_test_preds_list.append(y_probas)
            y_test_preds = y_probas.argmax(axis=1)
            print('    Sample predictions before inverse_transform: %s' %y_test_preds[:5])
            if cat_vocab_dict["target_transformed"]:
                try:
                    y_test_preds = cat_vocab_dict['target_le'].inverse_transform(y_test_preds)
                    print('    Sample predictions after inverse_transform: %s' %y_test_preds[:5])
                    y_test_preds_t = y_test_preds[:DS_LEN]
                    y_test_preds_list.append(y_test_preds_t)
                except:
                    print('    Inverse transform erroring. Continuing...')
                    y_test_preds_t = y_test_preds[:DS_LEN]
                    y_test_preds_list.append(y_test_preds_t)
            else:
                print('    Sample predictions after transform: %s' %y_test_preds[:5])
                y_test_preds_t = y_test_preds[:DS_LEN]
                y_test_preds_list.append(y_test_preds_t)
        else:
            #### This is for Single Label regression problems ######
            y_test_preds = y_probas.ravel()
            y_test_preds_t = y_test_preds[:DS_LEN]
            y_test_preds_list.append(y_test_preds_t)
    else:
        if modeltype == 'Regression':
            y_test_preds_list.append(y_probas)
            ### This is for Multi-Label Regresison problems ###
            for each_t in range(len(y_probas)):
                if each_t == 0:
                        y_test_preds = y_probas[each_t].mean(axis=1)
                else:
                        y_test_preds = np.c_[y_test_preds, y_probas[each_t].mean(axis=1)]
                y_test_preds_t = y_test_preds[:DS_LEN]
                y_test_preds_list.append(y_test_preds_t)
        else:
            y_preds = []
            #### This is Multi-Label Classification problems ######
            y_test_preds_list.append(y_probas)
            ### in Multi-Label predictions, output is a list if loading test datafile or dataframe ##
            if isinstance(y_probas, list):
                print('Multi-Label Predictions has %s outputs' %len(y_probas))
            else:
                print('Multi-Label Predictions shape:%s' %(y_probas.shape,))
            for each_t in range(len(y_probas)):
                y_test_preds = y_probas[each_t].argmax(axis=1)
                print('    Sample predictions for label: %s before transform: %s' %(each_t, y_test_preds[:5]))
                if cat_vocab_dict["target_transformed"]:
                    try:
                        y_test_preds = cat_vocab_dict[each_target]['target_le'].inverse_transform(
                                                y_test_preds)
                        print('    Sample predictions after inverse_transform: %s' %y_test_preds[:5])
                        if each_t == 0:
                            y_preds = y_test_preds
                        else:
                            y_preds = np.c_[y_preds, y_test_preds]
                        y_test_preds_t = y_preds[:DS_LEN,:]
                        y_test_preds_list.append(y_test_preds_t)
                    except:
                        print('    Inverse transform erroring. Continuing...')
                        if each_t == 0:
                            y_preds = y_test_preds
                        else:
                            y_preds = np.c_[y_preds, y_test_preds]
                        y_test_preds_t = y_preds[:DS_LEN]
                        y_test_preds_list.append(y_test_preds_t)
                else:
                    if each_t == 0:
                        y_preds = y_test_preds
                    else:
                        y_preds = np.c_[y_preds, y_test_preds]
                    y_test_preds_t = y_preds[:DS_LEN]
                    y_test_preds_list.append(y_test_preds_t)
    return y_test_preds_list
###########################################################################################
from PIL import Image
import numpy as np
from skimage import transform
def process_image_file(filename, img_height, img_weight, img_channels):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
##############################################################################################
def predict_images(test_image_dir, model_or_model_path, cat_vocab_dict, keras_model_type):
    project_name = cat_vocab_dict["project_name"]
    model_loaded, cat_vocab_dict = load_model_dict(model_or_model_path, cat_vocab_dict, project_name, keras_model_type)
    ##### Now load the classes neede for predictions ###
    y_test_preds_list = []
    classes = cat_vocab_dict['image_classes']
    img_height = cat_vocab_dict["image_height"]
    img_width = cat_vocab_dict["image_width"]
    batch_size = cat_vocab_dict["batch_size"]
    img_channels = cat_vocab_dict["image_channels"]
    if isinstance(test_image_dir, str):
        if test_image_dir.split(".")[-1] in ["jpg","png"]:
            print("    loading and predicting on file : %s" %test_image_dir)
            pred_label = model_loaded.predict(process_image_file(test_image_dir,
                                img_height, img_weight, img_channels))
            print('Predicted Label: %s' %(classes[np.argmax(pred_label)]))
            print('Predicted probabilities: %s' %pred_label)
        else:
            print("    loading and predicting on folder: %s" %test_image_dir)
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_image_dir,
                          seed=111,
                          image_size=(img_height, img_width),
                          batch_size=batch_size)
            y_probas = model_loaded.predict(test_ds)
            ### DS_LEN for image directories rarely exceeds 10000 so just set this as default ##
            DS_LEN = 10000
            pred_label = convert_predictions_from_model(y_probas, cat_vocab_dict, DS_LEN)
            return pred_label
    else:
        print('Error: test_image_dir should be either a directory containining test folder or a single JPG or PNG image file')
        return None
########################################################################################################
def predict_text(test_text_dir, model_or_model_path, cat_vocab_dict, keras_model_type):
    project_name = cat_vocab_dict["project_name"]
    model_loaded, cat_vocab_dict = load_model_dict(model_or_model_path, cat_vocab_dict, project_name, keras_model_type)
    ##### Now load the classes neede for predictions ###
    y_test_preds_list = []
    classes = cat_vocab_dict['text_classes']
    if isinstance(test_text_dir, str):
        if test_text_dir.split(".")[-1] in ["txt"]:
            try:
                batch_size = cat_vocab_dict["batch_size"]
            except:
                batch_size = 32
            print("    loading and predicting on TXT file : %s" %test_text_dir)
            pred_label = model_loaded.predict(test_text_dir)
            print('Predicted Label: %s' %(classes[np.argmax(pred_label)]))
            print('Predicted probabilities: %s' %pred_label)
        elif test_text_dir.split(".")[-1] in ["csv"]:
            print("    loading and predicting on CSV file : %s" %test_text_dir)
            test_ds, cat_vocab_dict = load_test_data(test_text_dir, project_name, cat_vocab_dict=cat_vocab_dict,
                                                             verbose=0)
            DS_LEN = cat_vocab_dict['DS_LEN']
            batch_size = cat_vocab_dict["batch_size"]
            try:
                num_steps = int(np.ceil(DS_LEN/batch_size))
            except:
                num_steps = 1
            #########  See if you can predict here if not return the null result #####
            print('    number of steps needed to predict: %d' %num_steps)
            y_probas = model_loaded.predict(test_ds, steps=num_steps)
            pred_label = convert_predictions_from_model(y_probas, cat_vocab_dict, DS_LEN)
            return pred_label
        else:
            try:
                batch_size = cat_vocab_dict["batch_size"]
            except:
                batch_size = 32
            print("    loading and predicting on folder: %s" %test_text_dir)
            test_ds = tf.keras.preprocessing.text_dataset_from_directory(test_text_dir,
                          seed=111,
                          batch_size=batch_size)
            y_probas = model_loaded.predict(test_ds)
            try:
                DS_LEN = cat_vocab_dict['DS_LEN']
            except:
                ### just set up a default number 10,000
                DS_LEN =  10000
            pred_label = convert_predictions_from_model(y_probas, cat_vocab_dict, DS_LEN)
            return pred_label
    else:
        print('Error: test_text_dir should be either a directory containining test folder or a single .txt file')
        return None
##########################################################################################################
