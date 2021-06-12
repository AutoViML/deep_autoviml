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
import os
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, CategoryCrossing
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, Discretization, Hashing
from tensorflow.keras.layers import Embedding, Reshape, Dropout, Dense, GaussianNoise

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
#####################################################################################
# Utils
from deep_autoviml.utilities.utilities import print_one_row_from_tf_dataset, print_one_row_from_tf_label
from deep_autoviml.utilities.utilities import print_classification_metrics, print_regression_model_stats
from deep_autoviml.utilities.utilities import print_classification_model_stats, plot_history, plot_classification_results
from deep_autoviml.utilities.utilities import get_compiled_model, add_inputs_outputs_to_model_body
from deep_autoviml.utilities.utilities import check_if_GPU_exists
from deep_autoviml.utilities.utilities import save_valid_predictions, predict_plot_images

from deep_autoviml.data_load.extract import find_batch_size
from deep_autoviml.modeling.create_model import check_keras_options
from deep_autoviml.modeling.one_cycle import OneCycleScheduler
#####################################################################################
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
#### probably the most handy function of all!
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
##############################################################################################
import time
import os
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from collections import defaultdict
from tensorflow.keras import callbacks
#########################################################################################
### This is the Storm-Tuner which stands for Stochastic Random Mutator tuner
###  More details can be found in this github: https://github.com/ben-arnao/stochasticmutatortuner
###   You can also pip install storm-tuner --upgrade to get the latest version ##########
#########################################################################################
from storm_tuner import Tuner
def train_image_model(deep_model, train_ds, valid_ds, cat_vocab_dict, 
                      keras_options, project_name, save_model_flag):
    epochs = check_keras_options(keras_options, "epochs", 20)
    print('Training image model. This will take time...')
    history = deep_model.fit(train_ds, epochs=epochs, validation_data=valid_ds)
    result = deep_model.evaluate(valid_ds)
    print('    Model accuracy in Image validation data: %s' %result[1])
    plot_history(history, "accuracy", 1)
    classes = cat_vocab_dict["image_classes"]
    predict_plot_images(deep_model, valid_ds, classes)
    save_model_path = os.path.join(project_name, "image_model")
    if save_model_flag:
        print('\nSaving model in %s now...this will take time...' %save_model_path)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        deep_model.save(save_model_path)
        cat_vocab_dict['saved_model_path'] = save_model_path
        print('     deep_autoviml image_model saved in %s directory' %save_model_path)
    else:
        print('\nModel not being saved since save_model_flag set to False...')
    return deep_model, cat_vocab_dict