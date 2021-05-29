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
from tensorflow import keras
#print('Tensorflow version on this machine: %s' %tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import layers
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
from tensorflow.keras.models import Model, load_model
#############################################################################################
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from IPython.core.display import Image, display
import pickle
#############################################################################################
##### Suppress all TF2 and TF1.x warnings ###################
try:
    tf.logging.set_verbosity(tf.logging.ERROR)
except:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
############################################################################################
from tensorflow.keras.layers import Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
############################################################################################
import time
import os
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from collections import defaultdict
############################################################################################
# data pipelines 
from .data_load.classify_features import classify_features
from .data_load.classify_features import classify_features_using_pandas

from .data_load.classify_features import EDA_classify_and_return_cols_by_type
from .data_load.classify_features import EDA_classify_features
from .data_load.extract import find_problem_type, transform_train_target
from .data_load.extract import load_train_data, load_train_data_file
from .data_load.extract import load_train_data_frame

# keras preprocessing
from .preprocessing.preprocessing import perform_preprocessing
from .preprocessing.preprocessing_tabular import preprocessing_tabular
from .preprocessing.preprocessing_nlp import preprocessing_nlp

# keras models and bring-your-own models
from .modeling.create_model import create_model
from .models import basic, deep, big_deep, giant_deep, cnn1, cnn2
from .modeling.train_model import train_model
from .modeling.train_custom_model import train_custom_model, return_optimizer
from .modeling.predict_model import predict

# Utils
from .utilities.utilities import print_one_row_from_tf_dataset
from .utilities.utilities import print_one_row_from_tf_label
from .utilities.utilities import get_compiled_model, add_inputs_outputs_to_model_body
from .utilities.utilities import check_if_GPU_exists

#############################################################################################
#### probably the most handy function of all!  ###############################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
##############################################################################################
def fit(train_data_or_file, target, keras_model_type="basic", project_name="deep_autoviml", 
                                save_model_flag=True, model_options={},
                                keras_options={}, use_my_model='', verbose=0):
    """
    ####################################################################################
    ####                          Deep AutoViML                                     ####
    ####                       Developed by Ram Seshadri (2021)                     ####
    ####                      Python 3, Tensforflow >= 2.4                          ####
    ####################################################################################
    Inputs:
    train_data_or_file: can be file or pandas dataframe: you need to give path to filename.
    target: string or list. You can give one variable (string) or multiple variables (list)
    keras_model_type: default = 'Deep', 'Big Deep', 'Giant Deep', 'Custom', 'CNN1',
                         'CNN2'. <more to come>
    project_name: default = "deep_autoviml". This is used to name the folder to save model.
    save_model_flag: default = False: it determines wher you want to save your trained model 
                    to local drive. If True, it will save it locally in project_name folder.
    use_my_model: default = '' - you can create a file with any model architecture you 
                    want and send in name of that file here. We will import that model 
                    file and use it as  model to run with  inputs and output pipeline 
                    we create. You can name  file anything you want but Don't name 
                    your model file as tensorflow.py or keras.py since when we import 
                    that file, it will overwrite tensorflow and keras functions in
                     your code (disaster!) Also, you must name  model variable as "model" 
                     in that file. So that way, when we import it, we will use it as
                      "import model from xyz" file. Important!
                    Additionally, you can create a Sequential model variable and send it.
    keras_options: dictionary:  you can send in any keras model option you want: optimizer, 
                epochs, batchsize, etc.
            "batchsize": default = "": you can leave it blank and we will automatically 
                calculate a batchsize
            "patience": default = 10 ### patience of 10 seems ideal. You can raise or lower it
            "epochs": default = 500 ## 500 seems ideal for most scenarios ####
            "steps_per_epoch": default = 5 ### 5 seems ideal for most scenarios 
            'optimizer': default = RMSprop(lr=0.1, rho=0.9) ##Adam(lr=0.1)   #SGD(lr=0.1)
            'kernel_initializer': default =  'glorot_uniform' ### Others:  'he_uniform', etc.
            'num_layers': default = 2 : # this defines  number of layers if you choose custom model
            'loss': default = it will choose automatically based on modeltype
                    ### you can define any keras loss function such as mae, mse, etc.
            'metrics': default = it will choose automatically based on modeltype 
                    ##  you can define any keras metric you like
            'monitor': default = it will choose automatically based on modeltype
            'mode': default = it will choose automatically based on modeltype
            "lr_scheduler": default = "onecycle" but you can choose from any below:
                    ##  ["scheduler", 'onecycle', 'rlr' (reduce LR on plateau), 'decay']
            "early_stopping": default = False. You can change it to True.
    model_options: dictionary:  you can send in any deep autoviml model option you 
                    want to change using this dictionary.
            You can change  following as long as you use this option and  same exact wordings:
            For example: let's say you want to change  number of categories in a variable 
                        above which it is not a cat variable.
            You can change that using  following option:
                model_options_defaults["variable_cat_limit"] = 30
            Similarly for the number of characters above which a string variable will be  
                considered an NLP variable: model_options_defaults["nlp_char_limit"] = 30
            Another option would be to inform autoviml about  encoding in  CSV file for it to 
                    read such as 'latin-1' by setting {"csv_encoding": 'latin-1'}
            Other examples:
            "csv_encoding": default =  'utf-8'
            "cat_feat_cross_flag": if you want to cross categorical features such as A*B, B*C...
            "sep" : default = "," comma but you can override it. Separator used in read_csv.
            "idcols": default: empty list. Specify which variables you want to exclude from model.
            "modeltype": default = '': if you leave it blank we will automatically determine it.
                    If you want to override, your options are: 'Regression', 'Classification', 
                    'Multi_Classification'.
                    We will figure out single label or multi-label problem based on your target 
                            being string or list.
            "header": default = 0 ### this is the header row for pandas to read
            "max_trials": default = 10 ## number of Storm Tuner trials ###
            "tuner": default = 'storm'  ## Storm Tuner is the default tuner. Optuna is the other option.
            "embedding_size": default = 50 ## this is the NLP embedding size minimum
    verbose = 1 will give you more charts and outputs. verbose 0 will run silently 
                with minimal outputs.
    """
    my_strategy = check_if_GPU_exists(1)
    model_options_copy = copy.deepcopy(model_options)
    keras_options_copy = copy.deepcopy(keras_options)
    ########    B E G I N N I N G    O F    S T R A T E G Y    S C O P E    #####################
    #with my_strategy.scope():

    shuffle_flag = False
    ####   K E R A S    O P T I O N S   - THESE CAN BE OVERRIDDEN by your input keras_options dictionary ####
    keras_options_defaults = {}
    keras_options_defaults["batchsize"] = ""
    keras_options_defaults['activation'] = ''
    keras_options_defaults['save_weights_only'] = True
    keras_options_defaults['use_bias'] = True
    keras_options_defaults["patience"] = "" ### patience of 20 seems ideal.
    keras_options_defaults["epochs"] = "" ## 500 seems ideal for most scenarios ####
    keras_options_defaults["steps_per_epoch"] = "" ### 10 seems ideal for most scenarios 
    keras_options_defaults['optimizer'] = ""
    keras_options_defaults['kernel_initializer'] =  '' 
    keras_options_defaults['num_layers'] = ""
    keras_options_defaults['loss'] = ""
    keras_options_defaults['metrics'] = ""
    keras_options_defaults['monitor'] = ""
    keras_options_defaults['mode'] = ""
    keras_options_defaults["lr_scheduler"] = ""
    keras_options_defaults["early_stopping"] = ""

    list_of_keras_options = ["batchsize", "activation", "save_weights_only", "use_bias",
                            "patience", "epochs", "steps_per_epoch", "optimizer",
                            "kernel_initializer", "num_layers",
                            "loss", "metrics", "monitor","mode", "lr_scheduler","early_stopping"]

    keras_options = copy.deepcopy(keras_options_defaults)
    if len(keras_options_copy) > 0:
        print('Using following keras_options given as input:')
        for key in list_of_keras_options:
            if key in keras_options_copy.keys():
                print('    %s : %s' %(key, keras_options_copy[key]))
                keras_options[key] = keras_options_copy[key]

    list_of_model_options = ["idcols","modeltype","sep","cat_feat_cross_flag", "model_use_case",
                            "nlp_char_limit", "variable_cat_limit", "csv_encoding", "header",
                            "max_trials","tuner", "embedding_size"]

    model_options_defaults = defaultdict(str)
    model_options_defaults["idcols"] = []
    model_options_defaults["modeltype"] = ''
    model_options_defaults["sep"] = ","
    model_options_defaults["cat_feat_cross_flag"] = False
    model_options_defaults["model_use_case"] = ''
    model_options_defaults["nlp_char_limit"] = 30
    model_options_defaults["variable_cat_limit"] = 30
    model_options_defaults["csv_encoding"] = 'utf-8'
    model_options_defaults["header"] = 0 ### this is the header row for pandas to read
    model_options_defaults["max_trials"] = 10 ## number of Storm Tuner trials ###
    model_options_defaults['tuner'] = 'storm'  ## Storm Tuner is the default tuner. Optuna is the other option.
    model_options_defaults["embedding_size"] = 50 ## this is the NLP embedding size minimum
    
    model_options = copy.deepcopy(model_options_defaults)
    if len(model_options_copy) > 0:
        print('Using following model_options given as input:')
        for key in list_of_model_options:
            if key in model_options_copy.keys():
                print('    %s : %s' %(key, model_options_copy[key]))
                model_options[key] = model_options_copy[key]

    BUFFER_SIZE = int(1e4)

    idcols = model_options["idcols"]
    modeltype = model_options["modeltype"]
    sep = model_options["sep"]
    csv_encoding = model_options["csv_encoding"]
    model_use_case = model_options["model_use_case"]
    nlp_char_limit = model_options["nlp_char_limit"]
    cat_feat_cross_flag = model_options["cat_feat_cross_flag"]
    variable_cat_limit = model_options["variable_cat_limit"]
    header = model_options["header"]
    max_trials = model_options["max_trials"]

    print("""
#################################################################################
###########     L O A D I N G    D A T A    I N T O   TF.DATA.DATASET H E R E  #
#################################################################################
        """)
    dft, model_options, batched_data, var_df, cat_vocab_dict, keras_options = load_train_data(
                                            train_data_or_file, target, project_name, keras_options, 
                                            model_options, verbose=verbose)

    try:
        data_size = cat_vocab_dict['DS_LEN']
    except:
        data_size = 10000
        cat_vocab_dict['DS_LEN'] = data_size
        
    modeltype = model_options['modeltype']

    ##########  Perform keras preprocessing here by building all layers needed #############
    print("""
#################################################################################
###########     K E R A S     F E A T U R E    P R E P R O C E S S I N G  #######
#################################################################################
        """)

    nlp_inputs, meta_inputs, meta_outputs = perform_preprocessing(batched_data, var_df, 
                                                cat_vocab_dict, keras_model_type, 
                                                model_options, cat_feat_cross_flag,  
                                                                    verbose)


    if isinstance(model_use_case, str):
        if model_use_case:
            if model_use_case.lower() == 'pipeline':
                ##########  Perform keras preprocessing only and return inputs + keras layers created ##
                print('\nReturning a keras pipeline so you can create your own Functional model.')
                return nlp_inputs, meta_inputs, meta_outputs
            #### There may be other use cases for model_use_case in future hence leave this empty for now #

    #### you must create a functional model here 
    print('\nCreating a new Functional model here...')
    print(''' 
#################################################################################
###########     C R E A T I N G    A    K E R A S       M O D E L    ############
#################################################################################
        ''')
    ######### this is where you get the model body either by yourself or sent as input ##
    ##### This takes care of providing multi-output predictions! ######
    inputs = nlp_inputs+meta_inputs
    model_body, keras_options =  create_model(use_my_model, inputs, meta_outputs, 
                                        keras_options, var_df, keras_model_type,
                                        model_options)
    
    ###########    C O M P I L E    M O D E L    H E R E         #############
    if not keras_model_type.lower() == 'auto':
        #### For every keras_model_type other than "auto" use this #################
        print('Loading a pre-built %s model...' %keras_model_type)
        final_outputs = add_inputs_outputs_to_model_body(model_body, inputs, meta_outputs)
        #### This final outputs is the one that is taken into final dense layer and compiled
        print('    %s model loaded successfully. Now compiling model...' %keras_model_type)
        ###### This is where you compile the model after it is built ###############
        num_classes = model_options["num_classes"]
        num_labels = model_options["num_labels"]
        modeltype = model_options["modeltype"]
        val_mode = keras_options["mode"]
        val_monitor = keras_options["monitor"]
        val_loss = keras_options["loss"]
        val_metrics = keras_options["metrics"]
        if not keras_options['optimizer']:
            ### if it is an empty string, use the default Adam optimizer
            optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        else:
            optimizer = return_optimizer(keras_options['optimizer'])
        if modeltype == 'Regression':
            num_predicts = 1*num_labels
            output_activation = "linear"
        elif modeltype == 'Classification':
            num_predicts = int(num_classes*num_labels)
            output_activation = "sigmoid"
        else:
            num_predicts = int(num_classes*num_labels)
            output_activation = "sigmoid"
        ######  You need to compile the non-auto models here ###############
        deep_model = get_compiled_model(inputs, final_outputs, output_activation, num_predicts, 
                                num_labels, model_body, optimizer, val_loss, val_metrics, dft.shape[1])
        if dft.shape[1] > 100:
            print('Too many columns to show model summary. Continuing...')
        else:
            print(deep_model.summary())
    else:
        ### For auto models we will add input and output layers later. See below... #########
        deep_model = model_body

    if dft.shape[1] <= 100 :
        plot_filename = 'deep_autoviml_'+project_name+'_'+keras_model_type+'_model_before.png'
        try:
            tf.keras.utils.plot_model(model = deep_model, to_file=plot_filename, dpi=72,
                            show_layer_names=True, rankdir="LR", show_shapes=True)
            display(Image(retina=True, filename=plot_filename))
            print('Model plot saved in file: %s' %plot_filename)
        except:
            print('Model plot not saved due to error. Continuing...')
    print("""
#################################################################################
###########     T R A I N I N G    K E R A S   M O D E L   H E R E      #########
#################################################################################
    """)
    if keras_model_type.lower() == 'auto':
        print('Building and training an automatic model using %s Tuner...' %model_options_defaults['tuner'])
        deep_model, cat_vocab_dict = train_custom_model(inputs, meta_outputs,
                                         batched_data, target, keras_model_type, keras_options, 
                                         model_options, var_df, cat_vocab_dict, project_name, 
                                            save_model_flag, use_my_model, verbose) 
    else:
        print('Training a %s model option...' %keras_model_type)
        deep_model, cat_vocab_dict = train_model(deep_model, batched_data, target, keras_model_type,
                        keras_options, model_options, var_df, cat_vocab_dict, project_name, save_model_flag, verbose) 
    ##########    E N D    O F    S T R A T E G Y    S C O P E   #############

    ##### Plot the model again after you have done model search #############
    if dft.shape[1] <= 100 :
        plot_filename = 'deep_autoviml_'+project_name+'_'+keras_model_type+'_model_after.png'
        try:
            tf.keras.utils.plot_model(model = deep_model, to_file=plot_filename, dpi=72,
                            show_layer_names=True, rankdir="LR", show_shapes=True)
            display(Image(retina=True, filename=plot_filename))
            print('Model plot saved in file: %s' %plot_filename)
        except:
            print('Model plot not saved due to error. Continuing...')

    distributed_values = (deep_model, cat_vocab_dict)
    return distributed_values
############################################################################################

