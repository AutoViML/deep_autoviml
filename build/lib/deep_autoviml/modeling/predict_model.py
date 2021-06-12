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
import pdb
import time
############################################################################################
def load_test_data(test_data_or_file, project_name, target="", cat_vocab_dict="",
                                                 verbose=0):
    ### load a small sample of data into a pandas dataframe ##
    
    if isinstance(test_data_or_file, str):
        test_small = pd.read_csv(test_data_or_file) ### this reads the entire file
    else:
        test_small = copy.deepcopy(test_data_or_file)
    filesize = test_small.shape[0]
    print('Loaded test data size: %d' %filesize)
    #### All column names in Tensorflow should have no spaces ! So you must convert them here!
    sel_preds = ["_".join(x.split(" ")) for x in list(test_small) ]
    if isinstance(target, str):
        target = "_".join(target.split(" "))
    else:
        target = ["_".join(x.split(" ")) for x in target ]
    print('    Modified column names to fit no-spaces-in-column-names rule in Tensorflow!')
    test_small.columns = sel_preds

    #### This means it is not a test dataset - hence it has target columns - load it too!
    if isinstance(target, str):
        if target == '':
            target_name = None
            usecols = []
        else:
            usecols = [target]
            target_name = copy.deepcopy(target)
    elif isinstance(target, list):
        #### then it is a multi-label problem
        target_name = target
        usecols = target
    else:
        print('Error: Target %s type not understood' %type(target))
        return
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
    ### classify variables using the small dataframe ##
    model_options = {}
    if no_cat_vocab_dict:
        model_options['DS_LEN'] = 10000  ### Just set some default #######
        var_df, cat_vocab_dict = classify_features_using_pandas(test_small, target=target_name, 
                                    model_options=model_options, verbose=verbose)
    else:
        model_options['DS_LEN'] = cat_vocab_dict['DS_LEN'] ### you need this to classify features
        var_df, _ = classify_features_using_pandas(test_small, target=target_name, 
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
                                               label_name=target_name,
                                               num_epochs = num_epochs,
                                               column_defaults=column_defaults,
                                               shuffle=False,
                                               num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ############### Do this only for Multi_Label problems ######
        if len(usecols) > 1:
            data_batches = data_batches.map(lambda x: split_combined_ds_into_two(x, usecols, preds))
    else:
        if test_small.isnull().sum().sum() > 0:
            test_small = fill_missing_values_for_TF2(test_small, var_df)
        
        drop_cols = cat_vocab_dict['columns_deleted']
        if len(drop_cols) > 0:
            print('    Dropping %s columns from dataset...' %drop_cols)
            test_small.drop(drop_cols, axis=1, inplace=True)            

        if isinstance(target, str):
            if target != '':
                labels = test_small[target]
                test_small.drop(target, axis=1, inplace=True)
                data_batches = tf.data.Dataset.from_tensor_slices((dict(test_small), labels))
            else:
                #print('\ntarget variable is blank - continuing')
                data_batches = tf.data.Dataset.from_tensor_slices(dict(test_small))
        elif isinstance(target, list):
            ##### For multi-label problems, you need to use dict of labels as well ###
            labels = test_small[target]
            test_small.drop(target, axis=1, inplace=True)
            data_batches = tf.data.Dataset.from_tensor_slices((dict(test_small), dict(labels)))
        else:
            data_batches = tf.data.Dataset.from_tensor_slices(dict(test_small))
        ### batch it if you are creating it from a dataframe
        data_batches = data_batches.batch(batch_size, drop_remainder=False).repeat()

    print('    test data loaded successfully.')

    if verbose >= 1:
        print_one_row_from_tf_dataset(data_batches)
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
def load_model_dict(model_or_model_path, cat_vocab_dict, project_name):
    start_time = time.time()
    if not cat_vocab_dict:
        ### No cat_vocab_dict is given. Hence you must load it from disk ###
        try:
            pickle_path = os.path.join(project_name, "cat_vocab_dict.pickle")
            print('\nLoading cat_vocab_dict file using pickle in %s...will take time...' %pickle_path)
            cat_vocab_dict = pickle.load(open(pickle_path,"rb"))
            print('    Loaded pickle file in %s' %pickle_path)
            modeltype = cat_vocab_dict['modeltype']
        except:
            print('Unable to load model and data artifacts cat_vocab_dictionary file. Returning...')
            return []
    else:
        ### cat_vocab_dictionary is given #####
            modeltype = cat_vocab_dict['modeltype']
    ### Check if model is available to be loaded #######
    if isinstance(model_or_model_path, str):
        try:
            if model_or_model_path == "":
                model_or_model_path = os.path.join(project_name, keras_model_type)
            print('\nLoading deep_autoviml model from %s folder...' %model_or_model_path)
            model = tf.keras.models.load_model(os.path.join(model_or_model_path))
            print('    time taken in mins for loading model = %0.0f' %((time.time()-start_time)/60))
        except Exception as error:
            print('Could not load given model.\nError: %s\n Please check your model path and try again.' %error)
            return
    else:
        print('\nUsing %s model provided as input...' %model_or_model_path)
        model = model_or_model_path
    print('Time taken to load saved model = %0.0f seconds' %((time.time()-start_time)))
    return model, cat_vocab_dict
###################################################################################################
def predict(model_or_model_path, project_name, test_dataset, 
                    keras_model_type, cat_vocab_dict=""):
    start_time2 = time.time()
    model, cat_vocab_dict = load_model_dict(model_or_model_path, cat_vocab_dict, project_name)
    ##### load the test data set here #######
    if isinstance(test_dataset, str):
        test_ds, cat_vocab_dict2 = load_test_data(test_dataset, project_name=project_name, 
                                target="", cat_vocab_dict=cat_vocab_dict)
        batch_size = cat_vocab_dict2["batch_size"]
        DS_LEN = cat_vocab_dict2["DS_LEN"]
        print("test data size = ",DS_LEN, ', batch_size = ',batch_size)
    elif isinstance(test_dataset, pd.DataFrame) or isinstance(test_dataset, pd.Series):
        test_ds, cat_vocab_dict2 = load_test_data(test_dataset, project_name=project_name,
                                target="", cat_vocab_dict=cat_vocab_dict)
        test_small = test_dataset
        batch_size = cat_vocab_dict2["batch_size"]
        DS_LEN = cat_vocab_dict2["DS_LEN"]
        print("test data size = ",DS_LEN, ', batch_size = ',batch_size)
    else:
        ### It must be a tf.data.Dataset hence just load it as is ####
        test_ds = test_dataset
        DS_LEN = 100000
        batch_size = 64
        cat_vocab_dict2 = copy.deepcopy(cat_vocab_dict)
    ##### Now predict on the data set here ####
    ## num_steps is needed to predict on whole dataset once ##
    try:
        num_steps = int(np.ceil(DS_LEN/batch_size))
    except:
        num_steps = 1
    #########  See if you can predict here if not return the null result #####
    print('    number of steps needed to predict: %d' %num_steps)
    y_test_preds_list = []
    targets = cat_vocab_dict2['target_variables']
    
    try:
        y_probas = model.predict(test_ds, steps=num_steps)
        if len(targets) == 1:
            y_test_preds_list.append(y_probas)    
        else:
            y_test_preds_list = copy.deepcopy(y_probas)
    except Exception as error:
        print('Could not predict using given model and inputs.\nError: %s\n Please check your inputs and try again.' %error)
        return y_test_preds_list

    ##### Now you need to save the predictions ###
    modeltype = cat_vocab_dict2['modeltype']

    if len(targets) == 1:
        if modeltype != 'Regression':
            y_test_preds = y_probas.argmax(axis=1)
        else:
            if y_test.dtype == 'int':
                y_test_preds = y_probas.round().astype(int)
            else:
                y_test_preds = y_probas.ravel()
    else:
        if modeltype != 'Regression':
            #### This is for multi-label binary or multi-class problems ##
            for each_t in range(len(targets)):
                if each_t == 0:
                    y_test_preds = y_probas[each_t].argmax(axis=1).astype(int)
                else:
                    y_test_preds = np.c_[y_test_preds, y_probas[each_t].argmax(axis=1).astype(int)]
        else:
            ### This is for Multi-Label Regression ###
            for each_t in range(len(targets)):
                if each_t == 0:
                    y_test_preds = y_probas[each_t].mean(axis=1)
                else:
                    y_test_preds = np.c_[y_test_preds, y_probas[each_t].mean(axis=1)]
                if y_test.dtype == 'int':
                    y_test_preds = y_test_preds.round().astype(int)

    ##### Now you have to convert the output to original classes and labels ####
    
    num_labels = cat_vocab_dict['num_labels']
    num_classes = cat_vocab_dict['num_classes']
    if num_labels <= 1:
        #### This is for Single-Label Problems only ################################
        if modeltype == 'Regression':
            y_pred = copy.deepcopy(y_test_preds)
        else:
            labels = cat_vocab_dict['original_classes']
            if cat_vocab_dict['target_transformed']:
                target_names = cat_vocab_dict['transformed_classes']
                target_le = cat_vocab_dict['target_le']
                y_pred = y_probas.argmax(axis=1)
                y_pred = target_le.inverse_transform(y_pred)
            else:
                y_pred = y_probas.argmax(axis=1)
    else:
        if modeltype == 'Regression':
            #### This is for Multi-Label Regression ################################
            y_pred = copy.deepcopy(y_test_preds)
        else:
            #### This is for Multi-Label Classification ################################
            try:
                for i, each_target in enumerate(targets):
                    labels = cat_vocab_dict[each_target+'_original_classes']
                    if cat_vocab_dict['target_transformed']:
                        ###### Use a nice classification matrix printing module here #########
                        target_names = cat_vocab_dict[each_target+'_transformed_classes']
                        target_le = cat_vocab_dict['target_le'][i]
                        y_pred_trans = y_probas[i].argmax(axis=1)
                        y_pred_trans = target_le.inverse_transform(y_pred_trans)
                        if i == 0:
                            y_pred = copy.deepcopy(y_pred_trans)
                        else:
                            y_pred = np.c_[y_pred, y_pred_trans]
                    else:
                        y_pred_trans = y_probas[i].argmax(axis=1)
                        if i == 0:
                            y_pred = copy.deepcopy(y_pred_trans)
                        else:
                            y_pred = np.c_[y_pred, y_pred_trans]
            except:
                print('Error in inverse transforming predictions...')
                y_pred = y_test_preds

    #### save the predictions only upto input size ###
    if num_labels <= 1:
        y_test_preds = y_pred[:DS_LEN]
    else:
        y_test_preds = y_pred[:DS_LEN,:]

    ###### Now collect the predictions if there are more than one target ###
    y_test_preds_list.append(y_test_preds)

    #####  We now show how many items are in the output  ###################
    print('Returning model predictions in form of a list...of length %d' %len(y_test_preds_list))
    print('Time taken in mins for predictions = %0.0f' %((time.time()-start_time2)/60))
    return y_test_preds_list
############################################################################################
def convert_predictions_from_model(y_probas, cat_vocab_dict):
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
                    y_test_preds_t = cat_vocab_dict['target_le'].inverse_transform(y_test_preds)
                    print('    Sample predictions after inverse_transform: %s' %y_test_preds_t[:5])
                    y_test_preds_list.append(y_test_preds_t)                
                except:
                    print('    Inverse transform erroring. Continuing...')
                    y_test_preds_list.append(y_test_preds)
            else:
                print('    Sample predictions after transform: %s' %y_test_preds[:5])
                y_test_preds_list.append(y_test_preds)
        else:
            #### This is for Single Label regression problems ######
            y_test_preds = y_probas.ravel()
            y_test_preds_list.append(y_test_preds)
    else:
        if modeltype == 'Regression':
            y_test_preds_list.append(y_probas)
            ### This is for Multi-Label Regresison problems ###
            for each_t in range(len(y_probas)):
                if each_t == 0:
                        y_test_preds = y_probas[each_t].mean(axis=1)
                else:
                        y_test_preds = np.c_[y_test_preds, y_probas[each_t].mean(axis=1)]
                y_test_preds_list.append(y_test_preds)
        else:
            #### This is Multi-Label Classification problems ######
            y_test_preds_list.append(y_probas)
            print('Multi-Label Predictions shape:%s' %(y_probas.shape,))
            for each_t in range(len(y_probas)):
                y_test_preds_t = y_probas[each_t].argmax(axis=1)
                print('    Sample predictions for label: %s before transform: %s' %(each_t, y_test_preds_t[:5]))
                if cat_vocab_dict["target_transformed"]:
                    try:
                        y_test_preds_t = cat_vocab_dict[each_target]['target_le'].inverse_transform(
                                                y_test_preds_t)
                        print('    Sample predictions after inverse_transform: %s' %y_test_preds_t[:5])
                        y_test_preds_list.append(y_test_preds_t)                
                    except:
                        print('    Inverse transform erroring. Continuing...')
                        y_test_preds_list.append(y_test_preds_t)
                else:
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
def predict_images(test_image_dir, model_or_model_path, cat_vocab_dict):
    project_name = cat_vocab_dict["project_name"]
    model_loaded, cat_vocab_dict = load_model_dict(model_or_model_path, cat_vocab_dict, project_name)
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
            pred_label = convert_predictions_from_model(y_probas, cat_vocab_dict)
            return pred_label
    else:
        print('Error: test_image_dir should be either a directory containining test folder or a single JPG or PNG image file')
        return None
########################################################################################################