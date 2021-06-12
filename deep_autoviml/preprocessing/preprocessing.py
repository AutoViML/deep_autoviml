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
# data pipelines and feature engg here
from deep_autoviml.preprocessing.preprocessing_tabular import preprocessing_tabular
from deep_autoviml.preprocessing.preprocessing_nlp import preprocessing_nlp

# Utils
from deep_autoviml.utilities.utilities import get_model_defaults

############################################################################################
# TensorFlow ≥2.4 is required
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

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
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, concatenate, Embedding
from tensorflow.keras.layers import Reshape, Activation, Flatten

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from IPython.core.display import Image, display
import pickle

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
#### probably the most handy function of all!
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#############################################################################################
def perform_preprocessing(train_ds, var_df, cat_vocab_dict, keras_model_type,
                           keras_options, model_options, verbose=0):
    """
    Remember this is the most valuable part of this entire library!
    This is one humongous preprocessing step to build everything needed for preprocessing into a keras model!
    But it will break in some cases since we cannot handle every known dataset!
    It will be good enough for most instances to create a fast keras pipeline + baseline model. 
    You can always fine tune it.
    You can always create your own model and feed it once you have successfully created preprocessing pipeline.
    """
    num_classes = model_options["num_classes"]
    num_labels = model_options["num_labels"]
    modeltype = model_options["modeltype"]
    embedding_size = model_options["embedding_size"]
    cat_feat_cross_flag = model_options["cat_feat_cross_flag"]
    targets = cat_vocab_dict["target_variables"]

    batch_size = 32
    ##### set the defaults for the LSTM or GRU model here #########################
    # Convolution
    kernel_size = 3
    filters = 128
    pool_size = 4

    # LSTM
    lstm_output_size = 96
    gru_units = 96

    # Training
    drop_out = 0.2
    if modeltype == 'Regression':
        class_size = 1
    else:
        if num_classes == 2:
            class_size = 1
        else:
            class_size = num_classes    
    ###### Now calculate some layer sizes #####    
    data_size = cat_vocab_dict["DS_LEN"]
    if data_size <= 10000:
        dense_layer1 = max(32,int(data_size/300))
        dense_layer2 = max(16,int(dense_layer1*0.5))
    elif data_size > 10000 and data_size <= 1000000:
        dense_layer1 = max(96,int(data_size/500))
        dense_layer2 = max(64,int(dense_layer1*0.5))
    else:
        dense_layer1 = 128
        dense_layer2 = 64
    #################################################################################
    ###########     F E A T U R E    P R E P R O C E S S I N G   H E R E      #######
    #################################################################################
    nlps = var_df['nlp_vars']
    keras_options, model_options, num_predicts, output_activation = get_model_defaults(keras_options, 
                                    model_options, targets)
    ##################  NLP Text Features are Proprocessed Here  ################
    nlp_inputs = []
    nlp_names = []
    if len(nlps) > 0:
        print('Starting NLP string column layer preprocessing...')
        nlp_inputs, embedding, nlp_names = preprocessing_nlp(train_ds, model_options,
                                                var_df, cat_vocab_dict, 
                                                keras_model_type, verbose)
        print('    NLP Preprocessing completed.')
    else:
        print('There are no NLP variables in this dataset for preprocessing...')
    ##################  All other Features are Proprocessed Here  ################
    meta_outputs = []
    print('Preprocessing non-NLP layers for %s Keras model...' %keras_model_type)
    if not keras_model_type.lower() in ['deep_and_wide','deep and wide', 'deep wide','deep_wide', 'fast', 'fast1', 'fast2']:
        print('    starting categorical, float and integer layer preprocessing...')
        
        meta_outputs, meta_inputs, meta_names = preprocessing_tabular(train_ds, var_df, 
                                                    cat_feat_cross_flag, model_options,
                                                    cat_vocab_dict, keras_model_type, verbose)
        print('    All Non-NLP feature preprocessing for %s completed.' %keras_model_type)

        ### this is the order in which columns have been trained ###
        final_training_order = nlp_names + meta_names
        ### find their dtypes - remember to use element_spec[0] for train data sets!
        ds_types = dict([(col_name, train_ds.element_spec[0][col_name].dtype) for col_name in final_training_order ])
        col_type_tuples = [(name,ds_types[name]) for name in final_training_order]
        print('Inferred column names, layers and types (double-check for duplicates and correctness!): \n%s' %col_type_tuples)
    ##### You need to send in the ouput from embedding layer to this sequence of layers ####
    fast_models = ['fast', 'fast1','fast2','deep_and_cross','deep and cross',
                        'deep cross','deep_and_wide','deep and wide','deep wide']
    nlp_outputs = []
    if len(nlps) > 0:
        if keras_model_type.lower() in ['bert','nlp','text', 'use']:
            ###### This is where you define the NLP Embedded Layers ########
            #x = layers.Dense(64, activation='relu')(embedding)
            #x = layers.Dense(32, activation='relu')(x)
            #nlp_outputs = layers.Dropout(0.2)(x)
            #nlp_outputs = layers.Dropout(0.2)(embedding)
            if isinstance(meta_outputs, list):
                #### if there are no other variables then leave it as embedding output
                nlp_outputs = embedding
            else:
                #### If there are other variables, then convert this embedding to an output
                nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(embedding)
        elif keras_model_type.lower() in ['auto']:
            x = layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embedding)
            x = layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)
            # = layers.Bidirectional(layers.LSTM(dense_layer1, dropout=0.3, recurrent_dropout=0.3,
            #                                    return_sequences=False, batch_size=batch_size,
            #                                    kernel_regularizer=regularizers.l2(0.01)))(x)
        elif keras_model_type.lower() in ['lstm']:
            x = layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(embedding)
            x = layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)
            # = layers.Bidirectional(layers.LSTM(dense_layer1, dropout=0.3, recurrent_dropout=0.3,
            #                                    return_sequences=False, batch_size=batch_size,
            #                                    kernel_regularizer=regularizers.l2(0.01)))(x)

        elif keras_model_type.lower() in ['cnn1']:
            # Conv1D + global max pooling
            x = Conv1D(dense_layer1, 14, name='cnn_dense1', padding="same", 
                                    activation="relu", strides=3)(embedding)
            x = GlobalMaxPooling1D()(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)
        elif keras_model_type.lower() in fast_models:
            # We add a vanilla hidden layer that's all
            x = GlobalAveragePooling1D()(embedding)
            x = Dense(dense_layer1, activation="relu")(x)
            x = Dropout(0.5)(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)
        elif keras_model_type.lower() in ['gru','cnn2']:
            #### Use this only for Binary-Class classification problems ########
            ####  LSTM with 1D convnet with maxpooling ########
            x = Conv1D(filters, 
                             kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1)(embedding)
            x = MaxPooling1D(pool_size=pool_size)(x)
            x = GRU(units=gru_units,  dropout=drop_out, recurrent_dropout=drop_out)(x)
            if modeltype == 'Regression':
                #nlp_outputs = Dense(class_size, activation='linear')(x)
                x = Dense(128, activation='relu')(x)
            else:
                #nlp_outputs = Dense(class_size, activation='sigmoid')(x)
                x = Dense(128, activation='relu')(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)
        else:
            #### Use this only for Multi-Class classification problems #########
            ####  CNN Model: create a 1D convnet with global maxpooling ########
            x = Conv1D(128, kernel_size, activation='relu')(embedding)
            x = MaxPooling1D(kernel_size)(x)
            x = Conv1D(128, kernel_size, activation='relu')(x)
            x = MaxPooling1D(kernel_size)(x)
            x = Conv1D(128, kernel_size, activation='relu')(x)
            x = GlobalMaxPooling1D()(x)
            x = Dense(128, activation='relu')(x)
            #nlp_outputs = Dense(class_size, activation='softmax')(x)
            nlp_outputs = layers.Dense(num_predicts, activation=output_activation)(x)

    if keras_model_type.lower() in fast_models:
        print('    All Non-NLP feature preprocessing for %s completed.' %keras_model_type)
        meta_inputs = []
        return nlp_inputs, meta_inputs, nlp_outputs
    else:
        #### This is only for non-deep_and_wide models and other kinds of models ##
        if isinstance(meta_outputs, list):
            ### if meta_outputs is a list, it means there is no int, float or cat variable in this data set
            print('There is no numeric or cat or int variables in this data set.')
            if isinstance(nlp_outputs, list):
                ### if NLP_outputs is a list, it means there is no NLP variable in the data set
                print('    There is no NLP variable in this data set. Returning')
            else:
                print('    Number of predictions = %s from NLP variable fed into model' %(nlp_outputs.shape,))
                consolidated_outputs = nlp_outputs
        else:
            print('    Shape of output from numeric+integer+cat variables before model training = %s' %(meta_outputs.shape,))
            if isinstance(nlp_outputs, list):
                ### if NLP_outputs is a list, it means there is no NLP variable in the data set
                print('    There is no NLP variable in this data set. Continuing...')
                #x = layers.concatenate([meta_outputs])
                consolidated_outputs = meta_outputs
            else:
                ### if NLP_outputs is NOT a list, it means there is some NLP variable in the data set
                print('    Number of predictions = %s from NLP variable fed into model' %(nlp_outputs.shape,))
                consolidated_outputs = layers.concatenate([nlp_outputs, meta_outputs])

        print('Shape of output from all preprocessing layers before model training = %s' %(consolidated_outputs.shape,))
        return nlp_inputs, meta_inputs, consolidated_outputs
##########################################################################################