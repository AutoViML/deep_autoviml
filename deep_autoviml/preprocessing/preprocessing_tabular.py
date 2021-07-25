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
########################################################################################################################
###### Many thanks to Hasan Rafiq for his excellent tutorial on Tensorflow pipelines where I learnt many helpful hints:
###### https://colab.research.google.com/gist/rafiqhasan/6f00aecf1feafd83ba9dfefef8907ee8/dl-e2e-taxi-dataset-tf2-keras.ipynb
###### Watch the entire video below on Srivatsan Srinivasan's excellent YouTube channel: AI Engineering   ##############
######                       https://youtu.be/wPri78CFSEw                                                 ##############
########################################################################################################################
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
########################################################################################################################
from tensorflow.keras.layers import Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, AveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
#### probably the most handy function of all!
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#############################################################################################################################
###### Many thanks to ML Design Patterns by Lak Lakshmanan which provided the following date-time TF functions below:
######    You can find more at : https://github.com/GoogleCloudPlatform/ml-design-patterns/tree/master/02_data_representation
############################################################################################################################
import datetime
def get_dayofweek(s):
    DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat','Sun']
    ts = parse_datetime(s)
    return DAYS[ts.weekday()]

def get_monthofyear(s):
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ts = parse_datetime(s)
    return MONTHS[ts.month-1]

def get_hourofday(s):
    ts = parse_datetime(s)
    return str(ts.hour)

@tf.function
def dayofweek(ts_in):
    """
    This function converts dayofweek as a number to a string such as 4 means Thursday in dayofweek format.
    """
    return tf.map_fn(
        lambda dayofweek_number: tf.py_function(get_dayofweek, inp=[dayofweek_number], Tout=tf.string),
        ts_in)

@tf.function
def hourofday(ts_in):
    """
    This function converts dayofweek as a number to a string such as 4 means Thursday in dayofweek format.
    """
    return tf.map_fn(
        lambda dayofweek_number: tf.py_function(get_hourofday, inp=[dayofweek_number], Tout=tf.string),
        ts_in)

@tf.function
def monthofyear(ts_in):
    """
    This function converts dayofweek as a number to a string such as 4 means Thursday in dayofweek format.
    """
    return tf.map_fn(
        lambda dayofweek_number: tf.py_function(get_monthofyear, inp=[dayofweek_number], Tout=tf.string),
        ts_in)


def parse_datetime(timestring):
    if type(timestring) is not str:
        timestring = timestring.numpy().decode('utf-8') # if it is a Tensor
    return pd.to_datetime(timestring, infer_datetime_format=True, errors='coerce')

##########################################################################################################
from itertools import combinations
from collections import defaultdict
import copy
import time
def preprocessing_tabular(train_ds, var_df, cat_feature_cross_flag, model_options, cat_vocab_dict,
                                keras_model_type,verbose=0):
    """
    ############################################################################################
    # This preprocessing layer returns a tuple (all_features, all_inputs) as arguments to create_model function
    # You must then create a Functional model by transforming all_features into outputs like this:
    # The final step in create_model will use all_inputs as inputs
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(all_inputs, output)
    ############################################################################################
    """
    start_time = time.time()
    drop_cols = var_df['cols_delete']
    #########  Now that you have the variable classification dictionary, just separate them out! ##
    cats = var_df['categorical_vars']  ### these are low cardinality vars - you can one-hot encode them ##
    high_string_vars = var_df['discrete_string_vars']  ## discrete_string_vars are high cardinality vars ## embed them!
    bools = var_df['bools']
    int_bools = var_df['int_bools']
    int_cats = var_df['int_cats'] + var_df['int_bools']
    ints = var_df['int_vars']
    floats = var_df['continuous_vars']
    nlps = var_df['nlp_vars']
    idcols = var_df['IDcols']
    dates = var_df['date_vars']
    lats = var_df['lat_vars']
    lons = var_df['lon_vars']
    matched_lat_lons = var_df['matched_pairs']
    floats = left_subtract(floats, lats+lons)

    #### These are the most important variables from this program: all inputs and outputs
    all_inputs = []
    all_encoded = []
    all_features = []
    all_input_names = []
    dropout_rate = 0.1 ### this is needed for feature crossing

    ### just use this to set the limit for max tokens for different variables ###
    ### we are setting the number of max_tokens to be 2X the number of tokens found in train
    max_tokens_zip = defaultdict(int)
    ##### Just combine Boolean and cat variables here to set the vocab ####
    cats_copy = copy.deepcopy(cats)
    if len(cats_copy) > 0:
        for each_name in cats_copy:
            max_tokens_zip[each_name] = cat_vocab_dict[each_name]['vocab'] ### just send vocab in
    high_cats_copy = copy.deepcopy(high_string_vars)
    if len(high_cats_copy) > 0:
        for each_name in high_cats_copy:
            max_tokens_zip[each_name] = cat_vocab_dict[each_name]['vocab'] ### just send vocab in
    copy_int_cats = copy.deepcopy(int_cats)
    if len(copy_int_cats) > 0:
        for each_int in copy_int_cats:
            max_tokens_zip[each_int] = cat_vocab_dict[each_int]['vocab'].tolist() ### just send vocab in
    copy_int_bools = copy.deepcopy(int_bools)
    if len(copy_int_bools) > 0:
        for each_int in copy_int_bools:
            max_tokens_zip[each_int] = cat_vocab_dict[each_int]['vocab'].tolist() ### just send vocab in
    copy_floats = copy.deepcopy(floats+bools)
    if len(copy_floats) > 0:
        for each_float in copy_floats:
            max_tokens_zip[each_float] = cat_vocab_dict[each_float]['vocab_min_var'] ### just send vocab as its a list
    copy_ints = copy.deepcopy(ints)
    if len(copy_ints) > 0:
        for each_int in copy_ints:
            max_tokens_zip[each_int] = int(1*(cat_vocab_dict[each_int]['size_of_vocab'])) ### size of vocab here
    if verbose > 1:
        print('    Number of categories in:')
        for each_name in max_tokens_zip.keys():
            if not each_name in high_cats_copy:
                print('        %s: %s' %(each_name, max_tokens_zip[each_name]))
            else:
                continue

    ####### CAVEAT : All the inputs and outputs should follow this same sequence below! ######
    all_date_inputs = []
    all_bool_inputs = []
    all_int_inputs = []
    all_int_cat_inputs = []
    all_cat_inputs = []
    all_num_inputs = []
    all_latlon_inputs = []
    ############## CAVEAT: The encoded outputs should follow the same sequence as inputs above!
    all_date_encoded = []
    all_bool_encoded = []
    all_int_bool_encoded = []
    all_int_encoded = []
    all_int_cat_encoded = []
    all_cat_encoded = []
    all_high_cat_encoded = []
    all_feat_cross_encoded = []
    all_num_encoded = []
    all_latlon_encoded = []
    lat_lon_paired_encoded = []
    cat_encoded_dict = dict([])
    cat_input_dict = dict([])
    date_input_dict = dict([])
    bool_input_dict = dict([])
    ###############################
    high_cats_alert = 50 ### set this number to alery you when a variable has high dimensions. Should it?
    hidden_units = [32, 32]  ## this is the number needed for feature crossing
    ####### We start creating variables encoding with date-time variables first ###########
    dates_copy = copy.deepcopy(dates)
    if len(dates) > 0:
        for each_date in dates_copy:
            #### You just create the date-time input only once and reuse the same input again and again
            date_input = keras.Input(shape=(1,), name=each_date, dtype="string")
            date_input_dict[each_date] = date_input
            all_date_inputs.append(date_input)
            all_input_names.append(each_date)
            try:
                ### for datetime strings, you need to split them into hour, day and month ######
                encoded_hour = encode_date_time_var_hourofday_categorical(date_input, each_date, train_ds)
                all_date_encoded.append(encoded_hour)
                if verbose:
                    print('    %s : after date-hour encoding shape: %s' %(each_date, encoded_hour.shape[1]))
                    if encoded_hour.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Date hourofday preprocessing erroring' %each_date)
            try:
                ### for datetime strings, you need to split them into hour, day and month ######
                encoded_day = encode_date_time_var_dayofweek_categorical(date_input, each_date, train_ds)
                all_date_encoded.append(encoded_day)
                if verbose:
                    print('    %s : after date-day encoding shape: %s' %(each_date, encoded_day.shape[1]))
                    if encoded_day.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Date dayofweek preprocessing erroring' %each_date)
            try:
                ### for datetime strings, you need to split them into hour, day and month ######
                encoded_month = encode_date_time_var_monthofyear_categorical(date_input, each_date, train_ds)
                all_date_encoded.append(encoded_month)
                if verbose:
                    print('    %s : after date-month encoding shape: %s' %(each_date, encoded_month.shape[1]))
                    if encoded_month.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Date dayofweek preprocessing erroring' %each_date)
            #### This is where you do the category crossing of hourofday and dayofweek first 24*7 bins

            try:
                encoded_hour_day = encode_cat_feature_crosses_numeric(encoded_day, encoded_hour, train_ds,
                                            bins_num=24*7)
                all_date_encoded.append(encoded_hour_day)
                if verbose:
                    print('    %s : after date-hour-day encoding shape: %s' %(each_date, encoded_hour_day.shape[1]))
                    if encoded_hour_day.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Date day-hour cross preprocessing erroring' %each_date)
            #### This is where you do the category crossing of dayofweek and monthofyear first 12*7 bins
            try:
                encoded_month_day = encode_cat_feature_crosses_numeric(encoded_month, encoded_day, train_ds,
                                            bins_num=12*7)
                all_date_encoded.append(encoded_month_day)
                if verbose:
                    print('    %s : after date-day-month encoding shape: %s' %(each_date, encoded_month_day.shape[1]))
                    if encoded_month_day.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Date month-day cross preprocessing erroring' %each_date)


    #####  If boolean variables exist, you must do this here ######

    if len(bools) > 0:
        bools_copy = copy.deepcopy(bools)
        try:
            for each_bool in bools_copy:
                #### You just create the boolean input  as float since we are converting it ###
                bool_input = keras.Input(shape=(1,), name=each_bool, dtype="float32")
                bool_input_dict[each_bool] = bool_input
                encoded = bool_input
            all_bool_encoded.append(encoded)
            all_bool_inputs.append(bool_input)
            all_input_names.append(each_bool)
            if verbose:
                print('    %s : after boolean encoding, is now float with  shape: %s' %(each_bool, encoded.shape[1]))
        except:
            print('    Error: Skipping %s since Keras Bolean preprocessing is erroring' %each_bool)

    ######  This is where we handle Boolean Integer variables - we just combine them ##################
    int_bools_copy = copy.deepcopy(int_bools)
    if len(int_bools_copy) > 0:
        for each_int in int_bools_copy:
            try:
                int_input = keras.Input(shape=(1,), name=each_int, dtype="int32")
                cat_input_dict[each_int] = int_input
                vocab = max_tokens_zip[each_int]
                layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=vocab,
                            mask_token=None, num_oov_indices=1, output_mode="int")
                # Convert the string input values into a one hot encoding.
                encoded = layer(int_input)
                all_int_inputs.append(int_input)
                all_int_bool_encoded.append(encoded)
                all_input_names.append(each_int)
                if verbose:
                    print('    %s number of categories = %d and after integer encoding shape: %s' %(each_int,
                                            len(max_tokens_zip[each_int]), encoded.shape[1]))
            except:
                print('    Error: Skipping %s since Keras Boolean Integer preprocessing erroring' %each_int)

    ######  This is where we handle high cardinality >50 categories integers ##################
    ints_copy = copy.deepcopy(ints)
    if len(ints_copy) > 0:
        for each_int in ints_copy:
            try:
                ### for integers that are very high cardinality, you can cut them down by half for starters
                if max_tokens_zip[each_int] <= 100:
                    nums_bin = max(5, int(max_tokens_zip[each_int]/10))
                elif max_tokens_zip[each_int] > 100 and max_tokens_zip[each_int] <= 1000:
                    nums_bin = max(10, int(max_tokens_zip[each_int]/10))
                else:
                    nums_bin = max(20, int(max_tokens_zip[each_int]/40))
                int_input = keras.Input(shape=(1,), name=each_int, dtype="int32")
                encoded = encode_any_integer_to_hash_categorical(int_input, each_int,
                                                                        train_ds, nums_bin)
                all_int_inputs.append(int_input)
                all_int_encoded.append(encoded)
                all_input_names.append(each_int)
                if verbose:
                    print('    %s number of categories = %d and bins = %d: after integer hash encoding shape: %s' %(each_int,
                                            max_tokens_zip[each_int], nums_bin, encoded.shape[1]))
                    if (encoded.shape[1] >= high_cats_alert) or (max_tokens_zip[each_int] >= high_cats_alert):
                        print('        Alert! excessive feature trap. Should this not be a float variable?? %s' %each_int)
            except:
                print('    Error: Skipping %s since Keras Integer preprocessing erroring' %each_int)


    ######  This is where we handle low cardinality <=50 categories integers ##################

    ints_cat_copy = copy.deepcopy(int_cats)
    if len(ints_cat_copy) > 0:
        for each_int in ints_cat_copy:
            try:
                int_input = keras.Input(shape=(1,), name=each_int, dtype="int32")
                cat_input_dict[each_int] = int_input
                vocab = max_tokens_zip[each_int]
                encoded = encode_integer_to_categorical_feature(int_input, each_int,
                                                                        train_ds, vocab)
                all_int_cat_inputs.append(int_input)
                all_int_cat_encoded.append(encoded)
                all_input_names.append(each_int)
                if verbose:
                    print('    %s number of categories = %d: after integer categorical encoding shape: %s' %(
                                        each_int, len(vocab), encoded.shape[1]))
                    if encoded.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Integer Categorical preprocessing erroring' %each_int)

    ##### All Discrete String and Categorical features are encoded as strings  ###########
    cats_copy = copy.deepcopy(cats)
    if len(cats_copy) > 0:
        for each_cat in cats_copy:
            if each_cat in lats+lons:
                continue    ### skip if these variables are already in another list
            try:
                cat_input = keras.Input(shape=(1,), name=each_cat, dtype="string")
                cat_input_dict[each_cat] = cat_input
                vocab = max_tokens_zip[each_cat]
                max_tokens = len(vocab)
                encoded = encode_string_categorical_feature_categorical(cat_input, each_cat,
                                                                     train_ds, vocab)
                all_cat_inputs.append(cat_input)
                all_cat_encoded.append(encoded)
                cat_encoded_dict[each_cat] = encoded
                all_input_names.append(each_cat)
                if verbose:
                    print('    %s number of categories = %d: after string to categorical encoding shape: %s' %(
                                        each_cat, max_tokens, encoded.shape[1]))
                    if encoded.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Categorical preprocessing erroring' %each_cat)

    ##### All Discrete String and Categorical features are encoded as strings  ###########
    high_cats_copy = copy.deepcopy(high_string_vars)
    if len(high_cats_copy) > 0:
        for each_cat in high_cats_copy:
            if each_cat in lats+lons:
                continue    ### skip if these variables are already in another list
            try:
                #### The next line is not a typo: this input should be left without shape. Be Careful!
                cat_input = keras.Input(shape=(1,), name=each_cat, dtype="string")
                vocabulary = max_tokens_zip[each_cat]
                encoded = encode_any_feature_to_embed_categorical(cat_input, each_cat,
                                                                     train_ds, vocabulary)
                all_cat_inputs.append(cat_input)
                all_high_cat_encoded.append(encoded)
                cat_encoded_dict[each_cat] = encoded
                all_input_names.append(each_cat)
                if verbose:
                    print('    %s : after high cardinality cat encoding shape: %s' %(each_cat, encoded.shape[1]))
                    if encoded.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras Discrete Strings (high cats) preprocessing erroring' %each_cat)

    ####  If the feature crosses for categorical variables are requested, then do this here ###
    if cat_feature_cross_flag:
        if isinstance(cat_feature_cross_flag, str):
            if cat_feature_cross_flag in ['cat','categorical']:
                cross_cats =  copy.deepcopy(cats)
            elif cat_feature_cross_flag in ['num', 'numeric']:
                cross_cats = int_cats+int_bools
            elif cat_feature_cross_flag in ['both', 'Both']:
                cross_cats = cats + int_cats + int_bools
        else:
            #### If it is true just perform string categorical crosses ###
            cross_cats =  copy.deepcopy(cats)
        if len(cross_cats) < 2:
            print('Feature crossing requested but not many cat or int-cat or int-bool variables in data')
        else:
            print('    Performing %s feature crossing using %d variables: \n    %s' %(cat_feature_cross_flag, len(cross_cats), cross_cats))
    ##### Now you perform crosses for each kind of crosses requested ####################
    if cat_feature_cross_flag and len(cross_cats) > 1:
        feat_cross_encoded = perform_feature_crossing(cat_input_dict, cross_cats, cats, floats, max_tokens_zip, verbose)
        all_feat_cross_encoded.append(feat_cross_encoded)
    else:
        print('    no feature crossing performed')
    ##################################################################################
    # Numerical features are treated as Numericals  ### this is a direct feed to the final layer ###
    nums_copy = left_subtract(floats,lats+lons)
    num_only_encoded = []

    if len(nums_copy) > 0:
        for each_num in nums_copy:
            try:
                num_input = keras.Input(shape=(1,), name=each_num, dtype="float32")
                ### Let's assume we don't do any encoding but use Normalization ##
                feat_mean = max_tokens_zip[each_num][0]
                feat_var = max_tokens_zip[each_num][1]
                normalizer = Normalization(mean=feat_mean, variance=feat_var)
                encoded = normalizer(num_input)
                #encoded = encode_numerical_feature_numeric(num_input, each_num, train_ds)
                all_num_inputs.append(num_input)
                all_num_encoded.append(encoded)
                num_only_encoded.append(encoded)
                all_input_names.append(each_num)
                print('    %s numeric column left as is for feature preprocessing' %each_num)
            except:
                print('    Error: Skipping %s since Keras Float preprocessing erroring' %each_num)


    # Latitude and Longitude Numerical features are Binned first and then Category Encoded #######
    lat_lon_paired_dict = dict([])
    #### Just remember that dtype of Input should match the dtype of the column! #####
    # Latitude and Longitude Numerical features are Binned first and then Category Encoded #######
    lat_lists = []
    lats_copy = copy.deepcopy(lats)
    if len(lats_copy) > 0:
        for each_lat in lats_copy:
            lat_lists += list(cat_vocab_dict[each_lat]['vocab'])
        lats_copy = copy.deepcopy(lats)
        for each_lat in lats_copy:
            try:
                bins_lat = pd.qcut(lat_lists, q=find_number_bins(cat_vocab_dict[each_lat]['vocab']),
                                   duplicates='drop', retbins=True)[1]
                ##### Now we create the inputs and the encoded outputs ######
                lat_lon_input = keras.Input(shape=(1,), name=each_lat, dtype="float32")
                all_latlon_inputs.append(lat_lon_input)
                lat_lon_encoded = encode_binning_numeric_feature_categorical(lat_lon_input, each_lat, train_ds,
                                                bins_lat=bins_lat,
                                                bins_num=len(bins_lat))
                all_latlon_encoded.append(lat_lon_encoded)
                lat_lon_paired_dict[each_lat] = lat_lon_encoded
                all_input_names.append(each_lat)
                if verbose:
                    print('    %s : after latitude binning encoding shape: %s' %(each_lat, lat_lon_encoded.shape[1]))
                    if lat_lon_encoded.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras latitudes var preprocessing erroring' %each_lat)

    lon_lists = []
    lons_copy = copy.deepcopy(lons)
    if len(lons_copy) > 0:
        for each_lon in lons_copy:
            lon_lists += list(cat_vocab_dict[each_lon]['vocab'])
        lons_copy = copy.deepcopy(lons)
        for each_lon in lons_copy:
            try:
                bins_lon = pd.qcut(lon_lists, q=find_number_bins(cat_vocab_dict[each_lon]['vocab']),
                                   duplicates='drop', retbins=True)[1]
                ##### Now we create the inputs and the encoded outputs ######
                lat_lon_input = keras.Input(shape=(1,), name=each_lon, dtype="float32")
                all_latlon_inputs.append(lat_lon_input)
                lat_lon_encoded = encode_binning_numeric_feature_categorical(lat_lon_input, each_lon, train_ds,
                                                bins_lat=bins_lon,
                                                bins_num=len(bins_lon))
                all_latlon_encoded.append(lat_lon_encoded)
                lat_lon_paired_dict[each_lon] = lat_lon_encoded
                all_input_names.append(each_lon)
                if verbose:
                    print('    %s : after longitude binning encoding shape: %s' %(each_lon, lat_lon_encoded.shape[1]))
                    if lat_lon_encoded.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping %s since Keras longitudes var preprocessing erroring' %each_lon)

    #### this is where you match the pairs of latitudes and longitudes to create an embedding layer
    if len(matched_lat_lons) > 0:
        matched_lat_lons_copy = copy.deepcopy(matched_lat_lons)
        for (lat_in_pair, lon_in_pair) in matched_lat_lons_copy:
            try:
                encoded_pair = encode_feature_crosses_lat_lon_numeric(lat_lon_paired_dict[lat_in_pair],
                                                              lat_lon_paired_dict[lon_in_pair],
                                                       dataset=train_ds, bins_lat=bins_lat)
                lat_lon_paired_encoded.append(encoded_pair)
                if verbose:
                    print('    %s + %s : after matched lat-lon crosses encoding shape: %s' %(lat_in_pair, lon_in_pair, encoded_pair.shape[1]))
                    if encoded_pair.shape[1] > high_cats_alert:
                        print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
            except:
                print('    Error: Skipping (%s, %s) since Keras lat-lon paired preprocessing erroring' %(lat_in_pair, lon_in_pair))

    #####  SEQUENCE OF THESE INPUTS AND OUTPUTS MUST MATCH ABOVE - we gather all outputs above into a single list
    all_inputs = all_bool_inputs + all_date_inputs + all_int_inputs + all_int_cat_inputs + all_cat_inputs + all_num_inputs + all_latlon_inputs
    all_encoded = all_bool_encoded + all_date_encoded+all_int_bool_encoded+all_int_encoded+all_int_cat_encoded+all_cat_encoded+all_feat_cross_encoded+all_num_encoded+all_latlon_encoded+lat_lon_paired_encoded
    all_low_cat_encoded = all_int_bool_encoded+all_int_encoded+all_int_cat_encoded  ## these are integer outputs ####
    all_numeric_encoded = all_bool_encoded + all_date_encoded+all_cat_encoded+all_high_cat_encoded+all_feat_cross_encoded+all_num_encoded+all_latlon_encoded+all_latlon_encoded+lat_lon_paired_encoded## these are all float ##
    ###### This is where we determine the size of different layers #########
    data_size = model_options['DS_LEN']
    if len(all_numeric_encoded) == 0:
        meta_numeric_len = 1
    elif len(all_numeric_encoded) == 1:
        meta_numeric_len = all_numeric_encoded[0].shape[1]
    else:
        meta_numeric_len = layers.concatenate(all_numeric_encoded).shape[1]
    data_dim = int(data_size*meta_numeric_len)
    if data_dim <= 1e6:
        dense_layer1 = max(16,int(data_dim/30000))
        dense_layer2 = max(8,int(dense_layer1*0.5))
        dense_layer3 = max(4,int(dense_layer2*0.5))
    elif data_dim > 1e6 and data_dim <= 1e8:
        dense_layer1 = max(24,int(data_dim/500000))
        dense_layer2 = max(12,int(dense_layer1*0.5))
        dense_layer3 = max(6,int(dense_layer2*0.5))
    elif data_dim > 1e8 or keras_model_type.lower() in ['big_deep', 'big deep']:
        dense_layer1 = 300
        dense_layer2 = 200
        dense_layer3 = 100
    dense_layer1 = min(300,dense_layer1)
    dense_layer2 = min(200,dense_layer2)
    dense_layer3 = min(100,dense_layer3)
    ############   D E E P   and   W I D E   M O D E L S   P R E P R O C E S S I N G ########
    ####    P R E P R O C E S S I N G   F O R  A L L   O T H E R   M O D E L S ########
    ####Concatenate all features( Numerical input )
    skip_meta_categ1 = False
    #concat_kernel_initializer = "glorot_uniform"
    concat_kernel_initializer = "he_normal"
    concat_activation = 'relu'
    concat_layer_neurons = dense_layer1

    ####Concatenate all categorical features( Categorical input ) #########
    if len(all_low_cat_encoded) == 0:
        skip_meta_categ1 = True
        meta_categ1 = None
    elif len(all_low_cat_encoded) == 1:
        meta_input_categ1 = all_low_cat_encoded[0]
        meta_categ1 = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_categ1)
        meta_categ1 = keras.layers.BatchNormalization()(meta_categ1)
        meta_categ1 = layers.Activation(concat_activation)(meta_categ1)
    else:
        meta_input_categ1 = layers.concatenate(all_low_cat_encoded)
        #WIDE - This Dense layer connects to input layer - Categorical Data
        meta_categ1 = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_categ1)
        meta_categ1 = keras.layers.BatchNormalization()(meta_categ1)
        meta_categ1 = layers.Activation(concat_activation)(meta_categ1)

    skip_meta_categ2 = False
    if len(all_high_cat_encoded) == 0:
        skip_meta_categ2 = True
        meta_categ2 = None
    elif len(all_high_cat_encoded) == 1:
        meta_input_categ2 = all_high_cat_encoded[0]
        meta_categ2 = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_categ2)
        meta_categ2 = layers.BatchNormalization()(meta_categ2)
        meta_categ2 = layers.Activation(concat_activation)(meta_categ2)
    else:
        meta_input_categ2 = layers.concatenate(all_high_cat_encoded)
        meta_categ2 = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_categ2)
        meta_categ2 = layers.BatchNormalization()(meta_categ2)
        meta_categ2 = layers.Activation(concat_activation)(meta_categ2)

    skip_meta_numeric = False
    if len(all_numeric_encoded) == 0:
        skip_meta_numeric = True
        meta_numeric = None
    elif len(all_numeric_encoded) == 1:
        meta_input_numeric = all_numeric_encoded[0]
        #meta_numeric = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_numeric)
        meta_numeric = layers.BatchNormalization()(meta_input_numeric)
        meta_numeric = layers.Activation(concat_activation)(meta_numeric)
    else:
        #### You must concatenate these encoded outputs before sending them out!
        #DEEP - This Dense layer connects to input layer - Numeric Data
        #meta_numeric = layers.BatchNormalization()(meta_input_numeric)
        meta_input_numeric = layers.concatenate(all_numeric_encoded)
        #meta_numeric = layers.Dense(concat_layer_neurons, kernel_initializer=concat_kernel_initializer)(meta_input_numeric)
        meta_numeric = layers.BatchNormalization()(meta_input_numeric)
        meta_numeric = layers.Activation(concat_activation)(meta_numeric)


    ####Concatenate both Wide and Deep layers
    #### in the end, you copy it into another variable called all_features so that you can easily remember name
    all_encoded_dict = list(zip([skip_meta_categ1, skip_meta_categ2, skip_meta_numeric],
                                  ['meta_categ1', 'meta_categ2', 'meta_numeric']))
    ######   This is how you concatenate the various layers ###############################
    concat_layers = []
    try:
        for (each_skip, each_encoded) in all_encoded_dict:
            # The order in which we feed the inputs is as follows: nlps + dates + ints + cats + floats + lat-lons
            if each_skip:
                ### This means that you must skip adding this layer ######
                continue
            else:
                #### This means you must add that layer ##########
                concat_layers.append(eval(each_encoded))
    except:
        print('    Error: preprocessing layers for %s models is erroring' %keras_model_type)

    if len(concat_layers) == 0:
        print('There are no cat, integer or float variables in this data set. Hence continuing...')
        all_features = []
    elif len(concat_layers) == 1:
        all_features = concat_layers[0]
    else:
        all_features = layers.concatenate(concat_layers)

    print('Time taken for preprocessing (in seconds) = %d' %(time.time()-start_time))
    return all_features, all_inputs, all_input_names

#############################################################################################
def find_number_bins(series):
    """
    Input can be a numpy array or pandas series. Otherwise it will blow up. Be careful!
    Returns the recommended number of bins for any Series in pandas
    Input must be a float or integer column. Don't send in alphabetical series!
    """
    try:
        num_of_quantiles = int(np.log2(series.nunique())+1)
    except:
        num_of_quantiles = max(2, int(np.log2(len(series)/5)))
    return num_of_quantiles
#############################################################################################
#####   Thanks to Francois Chollet for his excellent tutorial on Keras Preprocessing functions!
#####    https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
#####  Some of the functions below are derived from the tutorial. I have added many more here.
############################################################################################
def encode_numerical_feature_numeric(feature_input, name, dataset):
    """
    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    """
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature_input)
    return encoded_feature

###########################################################################################
def encode_binning_numeric_feature_categorical(feature, name, dataset, bins_lat, bins_num=10):
    """
    Inputs:
    ----------
    feature: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    CategoryEncoding output dtype is float32 even if output is binary or count.
    """
    # Create a StringLookup layer which will turn strings into integer indices
    index = Discretization(bin_boundaries = bins_lat)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # If the bin_boundaries are given, then no adapt is needed since it is already known.
    #index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(num_tokens=bins_num+1, output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

###########################################################################################
def encode_string_categorical_feature_categorical(feature_input, name, dataset, vocab):
    """
    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    When the output_mode = "binary" or "count", the output is in float otherwise it is integer.
    """
    extra_oov = 3
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup(max_tokens=None, num_oov_indices=extra_oov,
                        vocabulary=vocab, output_mode="count")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature_input)

    return encoded_feature

###########################################################################################
def encode_integer_to_categorical_feature(feature_input, name, dataset, vocab):
    """
    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    When the output_mode = "binary" or "count", the output is in float otherwise it is integer.
    """
    extra_oov = 3
    # Create a StringLookup layer which will turn strings into integer indices
    ### For now we will leave the max_values as None which means there is no limit.
    index = IntegerLookup(vocabulary=vocab, mask_token=None,
                        num_oov_indices=extra_oov, oov_token=-9999,
                        output_mode='int')

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature_input)

    return encoded_feature

###########################################################################################
def encode_cat_feature_crosses_numeric(encoded_input1, encoded_input2, dataset, bins_num=64):
    """
    This function does feature crosses of two categorical features sent in as encoded inputs.
    DO NOT SEND in RAW KERAS.INPUTs = they won't work here. This function takes those that are encoded.
    It then creates a feature cross, hashes the resulting categories and then category encodes them.
    The resulting output can be directly used an encoded variable for building pipelines.

    Inputs:
    ----------
    encoded_input1: This must be an encoded input - create a Keras.input variable first.
             Then do a StringLookup column on it and then a CategoryEncoding of it. Now you
             can feed that encoded variable into this as the first input.
    encoded_input1: This must be an encoded input - Similar to above: create a Keras.input variable first.
             Then do a StringLookup column on it and then a CategoryEncoding of it. Now you
             can feed that encoded variable into this as the second input.
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!
    bins_num: this is the number of bins you want to use in the hashing of the column
            Typically this can be 64. But you can make it smaller or larger.


    Outputs:
    -----------
    cat_cross_cat1_cat2: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) -  None indicates it is batched.
    CategoryEncoding output dtype is float32 even if output is binary or count.
    """
    ###########   Categorical cross of two categorical features is done here    #########
    cross_cat1_cat2 = tf.keras.layers.experimental.preprocessing.CategoryCrossing()(
                                                [encoded_input1, encoded_input2])
    hash_cross_cat1_cat2 = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=bins_num)(
                                                cross_cat1_cat2)
    #cat_cross_cat1_cat2 = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    #                                    num_tokens = bins_num)(hash_cross_cat1_cat2)

    # Cross to embedding
    cat_cross_cat1_cat2 = tf.keras.layers.Embedding(
                        (bins_num + 1) , 10) (hash_cross_cat1_cat2)
    cat_cross_cat1_cat2 = tf.reduce_sum(cat_cross_cat1_cat2, axis=-2)

    return cat_cross_cat1_cat2
###########################################################################################
def encode_feature_crosses_lat_lon_numeric(cat_pickup_lat, cat_pickup_lon, dataset, bins_lat):
    """
    This function does feature crosses of a paired latitude and logitude sent in as encoded inputs.
    DO NOT SEND in RAW KERAS.INPUTs = they won't work here. This function takes those that are encoded.
    It then creates a feature cross, hashes the resulting categories and then category encodes them.
    The resulting output can be directly used an encoded variable for building pipelines.

    Inputs:
    ----------
    cat_pickup_lat: This must be an encoded input - create a Keras.input variable first.
             Then do a Discretization column on it and then a CategoryEncoding of it. Now you
             can feed that encoded variable into this as the first input.
    cat_pickup_lon: This must be an encoded input - Similar to above: create a Keras.input variable first.
             Then do a Discretization column on it and then a CategoryEncoding of it. Now you
             can feed that encoded variable into this as the second input.
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!
    bins_lat: this is a pandas qcut bins - DO NOT SEND IN A NUMBER. It will fail!
            Typically you do this after binning the Latitude or Longitude after pd.qcut and set ret_bins=True.


    Outputs:
    -----------
    embed_cross_pick_lon_lat: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, embedding_dim) - None indicates it is batched.
    hashing creates integer outputs. So don't concatenate it with float outputs.
    """
    num_bins_lat = len(bins_lat)
    if num_bins_lat <= 100:
        nums_bin = max(5, int(num_bins_lat/10))
    elif num_bins_lat > 100 and num_bins_lat <= 1000:
        nums_bin = max(10, int(num_bins_lat/10))
    else:
        nums_bin = max(20, int(nums_bin/40))
    ### nums_bin = (len(bins_lat) + 1) ** 2 ## this was the old one
    ###########   Categorical cross of two categorical features is done here    #########
    cross_pick_lon_lat = tf.keras.layers.experimental.preprocessing.CategoryCrossing()(
                            [cat_pickup_lat, cat_pickup_lon])
    hash_cross_pick_lon_lat = tf.keras.layers.experimental.preprocessing.Hashing(
                            num_bins=nums_bin)(cross_pick_lon_lat)

    # Cross to embedding
    embed_cross_pick_lon_lat = tf.keras.layers.Embedding(
                        nums_bin, 10) (hash_cross_pick_lon_lat)
    embed_cross_pick_lon_lat = tf.reduce_sum(embed_cross_pick_lon_lat, axis=-2)

    return embed_cross_pick_lon_lat
################################################################################
def encode_any_integer_to_hash_categorical(feature_input, name, dataset, bins_num=30):
    """
    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!
    bins_num: this is the number of bins you want the hashing layer to split the data into

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
            The Tensor has a shape of (None, bins_num) - None indicates data has been batched
    hashing always gives you int64 encodings. So make sure you concatenate it with other integer encoders.
    """
    # Use the Hashing layer to hash the values to the range [0, 30]
    hasher = Hashing(num_bins=bins_num, salt=1337)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    hasher.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = hasher(feature_input)

    return encoded_feature

###########################################################################################
def encode_any_feature_to_embed_categorical(feature_input, name, dataset, vocabulary):
    """
    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             column in your dataset that want to transform. Please make sure it has a
             shape of (None, 1).
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!
    vocabulary: this is the number of bins you want the hashing layer to split the data into

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
            The Tensor has a shape of (None, bins_num) - None indicates data has been batched
    """
    extra_oov = 4
    # Use the lookup table first and then use embedding layer
    lookup = StringLookup(
        vocabulary=vocabulary,
        mask_token=None,
        num_oov_indices=extra_oov,
        max_tokens=None,
        output_mode="int"
    )
    # Convert the string input values into integer indices.
    #feature_ds = dataset.map(lambda x, y: x[name])
    #feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    #lookup.adapt(feature_ds)
    encoded_feature = lookup(feature_input)
    embedding_dims = int(math.sqrt(len(vocabulary)))
    # Create an embedding layer with the specified dimensions.
    embedding = tf.keras.layers.Embedding(
        input_dim=len(vocabulary)+extra_oov, output_dim=embedding_dims
    )
    # Convert the index values to embedding representations.
    encoded_feature = embedding(encoded_feature)
    encoded_feature = Flatten()(encoded_feature)

    return encoded_feature

###########################################################################################
def encode_date_time_var_dayofweek_categorical(feature_input, name, dataset):
    """
    This function will split the day of week from date-time column and create a new column.
    It will take a keras.Input variable as input and return a keras.layers variable as output.

    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             date-time column in your dataset that you want to transform. Please make sure it has a
             shape of (None, 1). It will split the hour of day from that column and create a new column.
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    CategoryEncoding output dtype is float32 even if output is binary or count.
    """
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: dayofweek(x[name]))
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature_input)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(num_tokens=8, output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

def encode_date_time_var_monthofyear_categorical(feature_input, name, dataset):
    """
    This function will split the month of year from date-time column and create a new column.
    It will take a keras.Input variable as input and return a keras.layers variable as output.

    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             date-time column in your dataset that you want to transform. Please make sure it has a
             shape of (None, 1). It will split the hour of day from that column and create a new column.
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    CategoryEncoding output dtype is float32 even if output is binary or count.
    """
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: monthofyear(x[name]))
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature_input)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(num_tokens=13, output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

def encode_date_time_var_hourofday_categorical(feature_input, name, dataset):
    """
    This function will split the hour of day from date-time column and create a new column.
    It will take a keras.Input variable as input and return a keras.layers variable as output.

    Inputs:
    ----------
    feature_input: must be a keras.Input variable, so make sure you create a variable first for the
             date-time column in your dataset that you want to transform. Please make sure it has a
             shape of (None, 1). It will split the hour of day from that column and create a new column.
    name: this is the name of the column in your dataset that you want to transform
    dataset: this is the variable holding the tf.data.Dataset of your data. Can be any kind of dataset.
            for example: it can be a batched or a prefetched dataset.
            Warning: You must be careful to set num_epochs when creating this dataset.
                   If num_epochs=None, this function will loop forever. If you set it to a number,
                   it will stop after that many epochs. So be careful!

    Outputs:
    -----------
    encoded_feature: a keras.Tensor. You can use this tensor in keras models for training.
               The Tensor has a shape of (None, 1) - None indicates that it has not been
    CategoryEncoding output dtype is float32 even if output is binary or count.
    """
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: hourofday(x[name]))
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature_input)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(num_tokens=25, output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature
################################################################################
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
######  This code is a modified version of keras.io documentation code examples ##########
######   https://keras.io/examples/structured_data/wide_deep_cross_networks/    ##########
##########################################################################################
def encode_auto_inputs(inputs, CATEGORICAL_FEATURE_NAMES, FLOATS, CATEGORICAL_FEATURES_WITH_VOCABULARY,
                         hidden_units, use_embedding=False):
    ### These are for string cat variables with small number of categories #############
    cat_encoded = []
    numeric_encoded = []
    text_encoded = []
    encoded_features = []

    for feature_name in inputs:
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        extra_oov = 3
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            cat_encoded.append('')
            cat_len = len(vocabulary)
            encoded_feature = inputs[feature_name]
            encoded_feature = tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=vocabulary, mask_token=None, oov_token = '~UNK~')(encoded_feature)
            cat_encoded[-1] = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
                            num_tokens = cat_len + 1)(encoded_feature)
        elif feature_name in FLOATS:
            ### you just ignore the floats in cross models ####
            numeric_encoded.append('')
            feat_mean = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][0]
            feat_var = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][1]
            normalizer = Normalization(mean=feat_mean, variance=feat_var)
            numeric_encoded[-1] = normalizer(inputs[feature_name])
        else:
            cat_encoded.append('')
            if len(vocabulary) > 100:
                print('    ALERT! Excessive feature dimension of %s. Should %s be a float variable?' %(
                                    len(vocabulary), feature_name))
                use_embedding = True
            lookup = IntegerLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=extra_oov,
                max_tokens=None,
                oov_token=-9999,
                output_mode="count" if use_embedding else "binary",
            )
            # Use the numerical features as-is.
            encoded_feature = inputs[feature_name]
            cat_encoded[-1] = lookup(encoded_feature)
    ##### This is where are float encoded features are combined ###
    ####Concatenate all features( Numerical input )
    encoded_features = layers.concatenate(cat_encoded+numeric_encoded)
    return encoded_features
##################################################################################################
import math
def encode_fast_inputs(inputs, CATEGORICAL_FEATURE_NAMES, FLOATS, CATEGORICAL_FEATURES_WITH_VOCABULARY,
                         use_embedding=False):
    encoded_features = []
    for feature_name in inputs:
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        extra_oov = 3
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token but expecting some out of vocabulary
            # (oov) tokens, we set mask_token to None and  num_oov_indices to extra_oov.
            if len(vocabulary) > 50:
                use_embedding = True
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=extra_oov,
                max_tokens=None,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                # Convert the string input values into integer indices.
                encoded_feature = inputs[feature_name]
                encoded_feature = lookup(encoded_feature)
                embedding_dims = int(math.sqrt(len(vocabulary)))
                # Create an embedding layer with the specified dimensions.
                embedding = layers.Embedding(
                    input_dim=len(vocabulary)+extra_oov, output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                encoded_feature = embedding(encoded_feature)
                encoded_feature = Flatten()(encoded_feature)
            else:
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(inputs[feature_name])
        elif feature_name in FLOATS:
            ### you just ignore the floats in cross models ####
            feat_mean = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][0]
            feat_var = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][1]
            normalizer = Normalization(mean=feat_mean, variance=feat_var)
            encoded_feature = normalizer(inputs[feature_name])
        else:
            if len(vocabulary) > 100:
                print('    ALERT! Excessive feature dimension of %s. Should %s be a float variable?' %(
                                    len(vocabulary), feature_name))
                use_embedding = True
            lookup = IntegerLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=extra_oov,
                max_tokens=None,
                oov_token=-9999,
                output_mode="count" if not use_embedding else "binary",
            )
            # Use the numerical features as-is.
            encoded_feature = inputs[feature_name]
            encoded_feature = lookup(encoded_feature)
        encoded_features.append(encoded_feature)
    ##### This is where are float encoded features are combined ###
    all_features = layers.concatenate(encoded_features)
    return all_features
##########################################################################################
######  This code is a modified version of keras.io documentation code examples ##########
######   https://keras.io/examples/structured_data/wide_deep_cross_networks/    ##########
##########################################################################################
def create_fast_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES, FLOATS):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in FLOATS:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.float32
            )
        elif feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.int32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.string
            )
    return inputs
#################################################################################
def create_all_inputs(FEATURE_NAMES, NUMERIC_FEATURE_NAMES, FLOATS):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in FLOATS:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.float32
            )
        elif feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(1,), dtype=tf.string
            )
    return inputs
########################################################################################
def encode_num_inputs(inputs, CATEGORICAL_FEATURE_NAMES, FLOATS, CATEGORICAL_FEATURES_WITH_VOCABULARY,
                         use_embedding=False):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in FLOATS:
            # Convert the string input values into a one hot encoding.
            encoded_feature = inputs[feature_name]
            encoded_features.append(encoded_feature)
    ##### This is where are float encoded features are combined ###
    all_features = layers.concatenate(encoded_features)
    return all_features
####################################################################################################
def encode_all_inputs(inputs, CATEGORICAL_FEATURE_NAMES, FLOATS, CATEGORICAL_FEATURES_WITH_VOCABULARY,
                         use_embedding=False):

    encoded_features = []
    for feature_name in inputs:
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        extra_oov = 3
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token but expecting some out of vocabulary
            # (oov) tokens, we set mask_token to None and  num_oov_indices to extra_oov.
            if len(vocabulary) > 50:
                use_embedding = True
            lookup = StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=extra_oov,
                max_tokens=None,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                # Convert the string input values into integer indices.
                encoded_feature = inputs[feature_name]
                encoded_feature = lookup(encoded_feature)
                embedding_dims = int(math.sqrt(len(vocabulary)))
                # Create an embedding layer with the specified dimensions.
                embedding = layers.Embedding(
                    input_dim=len(vocabulary)+extra_oov, output_dim=embedding_dims
                )
                # Convert the index values to embedding representations.
                encoded_feature = embedding(encoded_feature)
                encoded_feature = Flatten()(encoded_feature)
            else:
                # Convert the string input values into a one hot encoding.
                encoded_feature = lookup(inputs[feature_name])
        else:
            ##### This is for both integer and floats = they are considered same ###
            ### you just ignore the floats in cross models ####
            feat_mean = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][0]
            feat_var = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name][1]
            normalizer = Normalization(mean=feat_mean, variance=feat_var)
            encoded_feature = normalizer(inputs[feature_name])
            #encoded_feature = inputs[feature_name]
        encoded_features.append(encoded_feature)
    ##### This is where are float encoded features are combined ###
    all_features = layers.concatenate(encoded_features)
    return all_features
################################################################################
def perform_feature_crossing(cat_input_dict, cross_cats, cats, floats, max_tokens_zip, verbose=0):

    high_cats_alert = 50 ### set this number to alery you when a variable has high dimensions. Should it?
    hidden_units = [32, 32]  ## this is the number needed for feature crossing
    dropout_rate = 0.1 ### this is set at a low rate ###
    if len(cross_cats) > 20:
        print('    Too many categorical features (>20). hence taking first 20 features for crossing.')
        ls = cross_cats[:20]
    elif len(cross_cats) != len(cat_input_dict):
        ls = cross_cats
    else:
        ls = list(cat_input_dict.keys())
    ###################################################################################
    keys = list(cat_input_dict.keys())
    for each in keys:
        if each not in ls:
            cat_input_dict.pop(each)
    cross_cats = copy.deepcopy(ls)
    ##################################################################################
    try:
        # This is a deep and cross network for cat and int-cat + int-bool feature crosses
        each_cat_coded = encode_all_inputs(cat_input_dict, cats, floats, max_tokens_zip,
                                            use_embedding=True)
        cross = each_cat_coded
        for _ in hidden_units:
            units = cross.shape[-1]
            x = layers.Dense(units)(cross)
            if cross.dtype == x.dtype:
                cross = each_cat_coded * x + cross
            else:
                each_cat_coded = tf.cast(each_cat_coded, tf.float32)
                cross = tf.cast(cross, tf.float32)
                cross = each_cat_coded * x + cross

        cross = layers.BatchNormalization()(cross)

        deep = each_cat_coded
        for units in hidden_units:
            deep = layers.Dense(units)(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.ReLU()(deep)
            deep = layers.Dropout(dropout_rate)(deep)

        feat_cross_encoded = layers.concatenate([cross, deep])
        #feat_cross_encoded = cross
        if verbose:
            print('        feature crossing completed: cross encoding shape: %s' %(feat_cross_encoded.shape[1]))
            if feat_cross_encoded.shape[1] > high_cats_alert:
                print('        Alert! excessive feature dimension created. Check if necessary to have this many.')
    except:
        print('    Error: Skipping feature crossing since Keras preprocessing step is erroring')
    return feat_cross_encoded
########################################################################################
class OutputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OutputLayer, self).__init__()

    def call(self, inputs, **kwargs):
        outputs = tf.keras.backend.cast(inputs, "string")
        return outputs
##########################################################################################
def encode_bool_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        # Convert the string input values into a one hot encoding.
        lookup = StringLookup(
            vocabulary=['True', 'False'],
            mask_token=None,
            num_oov_indices=1,
            max_tokens=None,
            output_mode="binary"
        )
        encoded_feature = inputs[feature_name]
        encoded_feature = lookup(encoded_feature)
        encoded_features.append(encoded_feature)
    ##### This is where are float encoded features are combined ###
    if len(inputs) == 1:
        all_features = encoded_feature
    else:
        all_features = layers.concatenate(encoded_features)
    return all_features
############################################################################################
