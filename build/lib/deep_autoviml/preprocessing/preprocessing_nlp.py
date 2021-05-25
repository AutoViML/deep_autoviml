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

###########################################################################################
# We remove punctuations and HTMLs from tweets. This is done in a function,
# so that it can be passed as a parameter to the TextVectorization object.
import re
import string
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
##############################################################################################
def preprocessing_nlp(train_ds, model_options, var_df, cat_vocab_dict, embedding_size=100):
    """
    This produces a preprocessing layer for an incoming NLP column using TextVectorization from keras.
    You need to just send in a tf.data.DataSet from the training portion of your dataset and an nlp_column name.
    It will return a full-model-ready layer that you can add to your Keras Functional model as an NLP_layer!
    max_tokens_zip is a dictionary of each NLP column name and its max_tokens as defined by train data.
    """
    nlp_inputs = []
    all_nlp_encoded = []
    all_nlp_embeddings = []
    nlp_col_names = []
    nlp_columns = var_df['nlp_vars']

    #### max_tokens_zip calculate the max number of unique words in a vocabulary ####
    max_tokens_zip = defaultdict(int)
    cats_copy = copy.deepcopy(nlp_columns)
    if len(cats_copy) > 0:
        for each_name in cats_copy:
            max_tokens_zip[each_name] = cat_vocab_dict[each_name]['size_of_vocab']

    ###### Let us set up the defauls for embedding size and max tokens to process each column
    nlp_columns =  list(set(nlp_columns))
    nlp_columns_copy  = copy.deepcopy(nlp_columns)
    embedding_sizes = []
    for each_col in nlp_columns_copy:
        vocab_size = max_tokens_zip[each_col]
        embedding_sizes.append(min(embedding_size, vocab_size//200))
    print('    best embedding_size chosen = %s' %np.mean(embedding_sizes))
    for nlp_column in nlp_columns:
        #### Next, we add an NLP layer to map those vocab indices into a space of dimensionality
        #### Vocabulary size defines how many unique words you think might be in that training data
        ### Sequence length defines how we should convert each word into a sequence of integers of fixed length
        max_features = vocab_size = max_tokens_zip[nlp_column]
        embedding_dim = sequence_length = int(np.mean(embedding_sizes))

        # Use the text vectorization layer to normalize, split, and map strings to
        # integers. Note that the layer uses the custom standardization defined above.
        # Set maximum_sequence length as all samples are not of the same length.
        ### if you used custom_standardization function, you cannot load the saved model!! be careful!
        vectorize_layer = TextVectorization(
            #standardize=custom_standardization,
            standardize = 'lower_and_strip_punctuation',
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=sequence_length)
        #### Now let us process each column by using embeddings from Keras ####
        try:
            text_ds = train_ds.map(lambda x,y: x[nlp_column])

            # Tensorflow uses the word "adapt" to mean "fit" when learning vocabulary from a data set
            # You must call adapt first on a training data set and let it learn from that data set
            vectorize_layer.adapt(text_ds)

            # A string input for each string column ###############################
            nlp_input = tf.keras.Input(shape=(None,), dtype="string", name=nlp_column)

            ###### This is where you put NLP embedding layer into your data ####
            x = vectorize_layer(nlp_input)
            print(f"    {nlp_column} vocab size = {vocab_size}, embedding_dim={embedding_dim}, sample :\t{vectorize_layer.get_vocabulary()[:5]}")
            x = Embedding(max_features, embedding_dim, name=nlp_column+'_embedding')(x)
            #x = Flatten()(x)
            #x = Dropout(0.25)(x)  ### don't use Dropout after embedding - makes it less accurate!
            nlp_inputs.append(nlp_input)
            all_nlp_encoded.append(x)
            nlp_col_names.append(nlp_column)
        except:
            print('    Error: Skipping %s for keras layer preprocessing...' %nlp_column)

    ### we gather all outputs above into a single list here called all_features!
    if len(all_nlp_encoded) == 0:
        print('There are no NLP string variables in this dataset to preprocess!')
    elif len(all_nlp_encoded) == 1:
        all_nlp_embeddings = all_nlp_encoded[0]
    else:
        all_nlp_embeddings = layers.concatenate(all_nlp_encoded)

    return nlp_inputs, all_nlp_embeddings, nlp_col_names
###############################################################################################
def one_hot_encode_categorical_target(features, labels, categories):
    """Returns a one-hot encoded tensor representing categorical values."""
    # The entire encoding can fit on one line:
    labels = tf.cast(tf.equal(categories, tf.reshape(labels, [-1, 1])), tf.int32)
    return (features, labels)
##############################################################################################
def convert_classifier_targets(labels):
    """
    This handy function converts target labels that are binary or multi-class (whether integer or string) into integers.
    This is similar to a label encoder in scikit-learn but works on tensorflow tf.data.Datasets.
    Just send in a tf.data.Dataset and it will split it into features and labels and then turn them into correct labels.
    It returns the converted labels and a dictionary which you can use to convert it back to original labels. Neat!
    """
    _, converted_labels = tf.unique(labels)
    return converted_labels
#########################################################################################
def compare_two_datasets_with_idcol(train_ds, validate_ds, idcol,verbose=0):
    ls_test = list(validate_ds.as_numpy_iterator())
    ls_train = list(train_ds.as_numpy_iterator())
    if verbose >= 1:
        print('    Size of dataset 1 = %d' %(len(ls_train)))
        print('    Size of dataset 2 = %d' %(len(ls_test)))
    ts_list = [ls_test[x][0][idcol] for x in range(len(ls_test)) ]
    tra_list = [ls_train[x][0][idcol] for x in range(len(ls_train)) ]
    print('Alert! %d rows in common between dataset 1 and 2' %(len(tra_list) - len(left_subtract(tra_list, ts_list))))
##########################################################################################
def process_continuous_data(data):
    # Normalize data
    max_data = tf.reduce_max(data)
    min_data = tf.reduce_max(data)
    data = (tf.cast(data, tf.float32) - min_data)/(max_data - min_data)
    return tf.reshape(data, [-1, 1])
##########################################################################################
# Process continuous features.
def preprocess(features, labels):
    for feature in floats:
        features[feature] = process_continuous_data(features[feature])
    return features, labels
##########################################################################################
