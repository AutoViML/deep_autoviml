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
from itertools import combinations
from collections import defaultdict

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
############################################################################################
# data pipelines and feature engg here

# pre-defined TF2 Keras models and your own models here
from deep_autoviml.data_load.classify_features import check_model_options

# Utils

############################################################################################
# TensorFlow ≥2.4 is required
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, Hashing
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, CategoryEncoding, CategoryCrossing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization, Discretization
from tensorflow.keras.layers import Embedding, Flatten

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
import tensorflow_hub as hub
import tensorflow_text as text

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
def preprocessing_text(train_ds, keras_model_type, model_options):
    """
    ####################################################################################################
    This produces a preprocessing layer for an incoming NLP column using TextVectorization from keras.
    You need to just send in a tf.data.DataSet from training folder and it will automatically apply NLP.
    It will return a full-model-ready layer that you can add to your Keras Functional model as an NLP_layer!
    max_tokens_zip is a dictionary of each NLP column name and its max_tokens as defined by train data.
    ###########   Motivation and suggestions for coding for Image processing came from this blog #########
    Greatly indebted to Srivatsan for his Github and notebooks: https://github.com/srivatsan88/YouTubeLI
    ####################################################################################################
    """
    num_predicts = model_options["num_classes"]
    try:
        if keras_model_type.lower() in ["text"]:
            #######    L O A D     F E A T U R E    E X T R A C T O R   ################
            url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
            tf_hub_model = check_model_options(model_options, "tf_hub_model", url)
            feature_extractor_layer = hub.KerasLayer(tf_hub_model, output_shape=[20],
                                 input_shape=[], dtype=tf.string, trainable=False)
            units = 16
            print('Using Swivel-20D model from TensorFlow Hub')
        else:
            tf_hub_model = "https://tfhub.dev/google/nnlm-en-dim50/2"
            feature_extractor_layer = hub.KerasLayer(tf_hub_model, output_shape=[50],
                                 input_shape=[], dtype=tf.string, trainable=True)
            units = 32
            print('    Using NNLM-50D model from TensorFlow Hub')
        tf.random.set_seed(111)
        model = tf.keras.Sequential([
                  feature_extractor_layer,
                  tf.keras.layers.Dense(units, activation='relu'),
                  tf.keras.layers.Dense(num_predicts,activation='sigmoid')
                ])
        model.compile(
                  optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    except:
        print('    Error: Failed NLP preprocessing layer. Returning...')
        return
    return model
