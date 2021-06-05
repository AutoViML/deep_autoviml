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
import matplotlib.pyplot as plt
import tempfile
import pdb
import copy
import warnings
warnings.filterwarnings(action='ignore')
import functools
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
from collections import defaultdict
############################################################################################
# data pipelines and feature engg here
from deep_autoviml.models import basic, deep, big_deep, giant_deep, cnn1, cnn2
from deep_autoviml.preprocessing.preprocessing_tabular import encode_inputs, create_model_inputs

# Utils
from deep_autoviml.utilities.utilities import get_compiled_model, check_if_GPU_exists
from deep_autoviml.utilities.utilities import get_model_defaults, check_keras_options
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

from tensorflow.keras import layers
#######################################################################################
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.python.keras import backend as K
import sys
class BalancedAccuracy(tf.keras.metrics.Metric):
    """
    ##########################################################################################
    ###### Many thanks to the source below for this Balanced Accuracy Metric #################
    ###  https://github.com/saeyslab/DeepLearning_for_ImagingFlowCytometry/blob/master/model.py
    ##########################################################################################
    """
    def __init__(self, noc, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)

        self.noc = noc
        self.confusion_matrix = self.add_weight(
            name = "confusion_matrix",
            shape = (noc, noc),
            initializer = "zeros",
            dtype = tf.int32
        )

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(shape=v.get_shape())) for v in self.variables])

    def update_state(self, y_true, y_pred, sample_weight=None):
        confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=1), num_classes=self.noc)
        return self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        diag = tf.linalg.diag_part(self.confusion_matrix)
        rowsums = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        result = tf.math.reduce_mean(diag/rowsums, axis=0)
        return result
##########################################################################################
def create_model(use_my_model, inputs, meta_outputs, keras_options, var_df,
                        keras_model_type, model_options, cat_vocab_dict):
    """
    This is a handy function to create a Sequential model architecture depending on keras_model_type option given.
    It also can re-use a model_body (without input and output layers) given by the user as input for model_body.
    It returns a model_body as well as a tuple containing a number of parameters used on defining the model and training it.
    """
    data_size = model_options['DS_LEN']
    num_classes = model_options["num_classes"]
    num_labels = model_options["num_labels"]
    modeltype = model_options["modeltype"]
    patience = check_keras_options(keras_options, "patience", 10)
    cols_len = len([item for sublist in list(var_df.values()) for item in sublist])
    if not isinstance(meta_outputs, list):
        data_dim = int(data_size*meta_outputs.shape[1])
    else:
        data_dim = int(data_size*cols_len)
    #### These can be standard for every keras option that you use layers ######
    kernel_initializer = check_keras_options(keras_options, 'kernel_initializer', 'glorot_uniform')
    activation='relu'    
    
    ##############  S E T T I N G    U P  DEEP_WIDE, DEEP_CROSS, FAST MODELS    ########################
    cats = var_df['categorical_vars']  ### these are low cardinality vars - you can one-hot encode them ##
    high_string_vars = var_df['discrete_string_vars']  ## discrete_string_vars are high cardinality vars ## embed them!
    int_cats = var_df['int_cats']
    ints = var_df['int_vars']
    floats = var_df['continuous_vars']
    nlps = var_df['nlp_vars']

    FEATURE_NAMES = cats + high_string_vars + int_cats + ints + floats
    NUMERIC_FEATURE_NAMES = int_cats + ints + floats
    CATEGORICAL_FEATURE_NAMES = cats + high_string_vars
    
    vocab_dict = defaultdict(list)
    cats_copy = copy.deepcopy(CATEGORICAL_FEATURE_NAMES)
    if len(cats_copy) > 0:
        for each_name in cats_copy:
            vocab_dict[each_name] = cat_vocab_dict[each_name]['vocab']

    ######################   set some defaults for model parameters here ##############
    keras_options, model_options, num_predicts, output_activation = get_model_defaults(keras_options, model_options)
    ###################################################################################
    if data_dim <= 1e6:
        dense_layer1 = max(96,int(data_dim/30000))
        dense_layer2 = max(64,int(dense_layer1*0.5))
        dense_layer3 = max(32,int(dense_layer2*0.5))
    elif data_dim > 1e6 and data_dim <= 1e8:
        dense_layer1 = max(192,int(data_dim/50000))
        dense_layer2 = max(128,int(dense_layer1*0.5))
        dense_layer3 = max(64,int(dense_layer2*0.5))
    elif data_dim > 1e8 or keras_model_type == 'big_deep':
        dense_layer1 = 400
        dense_layer2 = 200
        dense_layer3 = 100
    dense_layer1 = min(300,dense_layer1)
    dense_layer2 = min(200,dense_layer2)
    dense_layer3 = min(100,dense_layer3)
    print('Recommended hidden layers (with units in each Dense Layer)  = (%d, %d, %d)\n' %(
                                dense_layer1,dense_layer2,dense_layer3))
    #### The Deep and Wide Model is a bit more complicated. So it needs some changes in inputs! ######
    if isinstance(use_my_model, str) :
        if use_my_model == "":
            if keras_model_type.lower() in ['basic', 'simple', 'default','simple_dnn','sample model']:
                ##########  Now that we have setup the layers correctly, we can build some more hidden layers
                model_body = basic.model
            elif keras_model_type.lower() in ['deep']:
                ##########  Now that we have setup the layers correctly, we can build some more hidden layers
                model_body = deep.model
            elif keras_model_type.lower() in ['big_deep', 'big deep']:
                ####################################################
                model_body = big_deep.model
            elif keras_model_type.lower() in ['giant_deep', 'giant deep']:
                ####################################################
                model_body = giant_deep.model
            elif keras_model_type.lower() in ['cnn1', 'cnn','cnn2']:
                ##########  Now that we have setup the layers correctly, we can build some more hidden layers
                # Conv1D + global max pooling
                if keras_model_type.lower() in ['cnn', 'cnn1']:
                    model_body = cnn1.model
                else:
                    model_body = cnn2.model
            elif keras_model_type.lower() in ['deep_and_wide','deep_wide','wide_deep', 
                                'wide_and_deep','deep wide', 'wide deep', 'fast', 'fast1']:
                ########## In case none of the options are specified, then set up a simple model!
                dropout_rate = 0.1
                #hidden_units = [32, 32]
                hidden_units = [dense_layer1, dense_layer2]
                all_inputs = create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES)
                wide = encode_inputs(all_inputs, CATEGORICAL_FEATURE_NAMES, vocab_dict,
                                use_embedding=False)
                wide = layers.BatchNormalization()(wide)
                deep = encode_inputs(all_inputs, CATEGORICAL_FEATURE_NAMES, vocab_dict,
                                use_embedding=True)
                keras_options, model_options, num_predicts, output_activation = get_model_defaults(keras_options, model_options)
                for units in hidden_units:
                    deep = layers.Dense(units)(deep)
                    deep = layers.BatchNormalization()(deep)
                    deep = layers.ReLU()(deep)
                    deep = layers.Dropout(dropout_rate)(deep)
                if len(nlps) > 0:
                    all_inputs = list(all_inputs.values()) ### convert input layers to a list
                    all_inputs += inputs
                    merged = layers.concatenate([meta_outputs, wide, deep])
                    final_outputs = layers.Dense(units=num_predicts, activation=output_activation)(merged)
                    model_body = keras.Model(inputs=all_inputs, outputs=final_outputs)
                else:
                    merged = layers.concatenate([wide, deep])
                    final_outputs = layers.Dense(units=num_predicts, activation=output_activation)(merged)
                    model_body = keras.Model(inputs=all_inputs, outputs=final_outputs)
                print('    Created deep and wide %s model, ...' %keras_model_type)
            elif keras_model_type.lower() in ['deep_and_cross', 'deep_cross', 'deep cross', 'fast2']:
                dropout_rate = 0.1
                #hidden_units = [32, 32]
                hidden_units = [dense_layer1, dense_layer2]
                all_inputs = create_model_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES)
                x0 = encode_inputs(all_inputs, CATEGORICAL_FEATURE_NAMES, vocab_dict,
                                use_embedding=True)
                cross = x0
                for _ in hidden_units:
                    units = cross.shape[-1]
                    x = layers.Dense(units)(cross)
                    cross = x0 * x + cross
                cross = layers.BatchNormalization()(cross)

                deep = x0
                for units in hidden_units:
                    deep = layers.Dense(units)(deep)
                    deep = layers.BatchNormalization()(deep)
                    deep = layers.ReLU()(deep)
                    deep = layers.Dropout(dropout_rate)(deep)
                merged = layers.concatenate([cross, deep])
                final_outputs = layers.Dense(units=num_predicts, activation=output_activation)(merged)
                model_body = keras.Model(inputs=all_inputs, outputs=final_outputs)
                print('    Created deep and cross %s model, ...' %keras_model_type)
                ################################################################################
            else:
                ### this means it's an auto model and you create one here 
                print('    building a %s model...' %keras_model_type)
                num_layers = check_keras_options(keras_options, 'num_layers', 1)
                model_body = tf.keras.Sequential([])
                for l_ in range(num_layers):
                    model_body.add(layers.Dense(dense_layer1, activation='selu', kernel_initializer="lecun_normal",
                                              activity_regularizer=tf.keras.regularizers.l2(0.01)))
        else:
            try:
                new_module = __import__(use_my_model)
                print('Using the model given as input to build model body...')
                model_body = new_module.model
                print('    Loaded model from %s file successfully...' %use_my_model)
            except:
                print('    Loading %s model is erroring, hence building a simple sequential model with one layer...' %keras_model_type)
                ########## In case none of the loading of files works, then set up a simple model!
                model_body = Sequential([layers.Dense(dense_layer1, activation='relu')])
    else:
        print('    Using your custom model given as input...')
        model_body = use_my_model
    ###########################################################################
    print('    %s model loaded and compiled successfully...' %keras_model_type)
    return model_body, keras_options
###############################################################################

