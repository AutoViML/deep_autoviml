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
# -*- coding: utf-8 -*-
################################################################################
#     deep_autoviml - build and test multiple Tensorflow 2.0 models and pipelines
#     Python v3.6+ tensorflow v2.4.1+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
import pandas as pd
import numpy as np
np.random.seed(99)
#### The warnings from Sklearn are so annoying that I have to shut it off #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn
####################################################################################
import re
import pdb
import pprint
from itertools import cycle, combinations
from collections import defaultdict, OrderedDict
import time
import sys
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
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

#######################################################################################################
def classify_features(dfte, depVar, model_options={}, verbose=0):
    max_cols_analyzed = 30
    dfte = copy.deepcopy(dfte)
    if isinstance(depVar, list):
        orig_preds = [x for x in list(dfte) if x not in depVar]
    else:
        orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(dfte[orig_preds], model_options, verbose)

    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    nlp_vars = var_df['nlp_vars']
    discrete_string_vars = var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    int_vars = var_df['int_vars'] + var_df['num_bool_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] +  var_df['string_bool_vars']
    date_vars = var_df['date_vars']
    continuous_vars = var_df['continuous_vars']
    ####### Now search for latitude and longitude variables ######
    lats, lons, matched_pairs = find_latitude_longitude_columns_in_df(dfte[orig_preds], verbose)
    if len(lats+lons) > 0:
        continuous_vars = left_subtract(continuous_vars, lats+lons)
        categorical_vars = left_subtract(categorical_vars, lats+lons)
        discrete_string_vars = left_subtract(discrete_string_vars, lats+lons)
    ######################################################################
    #cols_delete = list(set(IDcols+cols_delete))  ## leave IDcols in dataset. We need ID's to track rows.
    preds = [x for x in orig_preds if x not in cols_delete]

    var_df['cols_delete'] = cols_delete
    if len(cols_delete) == 0:
        print('        No variables removed since no ID or low-information variables found in data set')
    else:
        print('        %d variable(s) to be removed since they were ID or low-information variables'
                                %len(cols_delete))
        if verbose >= 1:
            print('    List of variables to be removed: %s' %cols_delete)
    #############  Check if there are too many columns to visualize  ################
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose > 1 and len(preds) <= max_cols_analyzed:
        marthas_columns(dfte,verbose)
        print("   Columns to delete:")
        ppt.pprint('   %s' % cols_delete)
        print("   Categorical variables: ")
        ppt.pprint('   %s' % categorical_vars)
        print("   Continuous variables:" )
        ppt.pprint('   %s' % continuous_vars)
        print("   Discrete string variables: " )
        ppt.pprint('   %s' % discrete_string_vars)
        print("   NLP string variables: " )
        ppt.pprint('   %s' % nlp_vars)
        print("   Date and time variables: " )
        ppt.pprint('   %s' % date_vars)
        if len(lats) > 0:
            print("   Latitude variables:" )
            ppt.pprint('   %s' % lats)
        if len(lons) > 0:
            print("   Longitude variables:" )
            ppt.pprint('   %s' % lons)
        if len(matched_pairs) > 0:
            print("   Matched Latitude and Longitude variables:" )
            ppt.pprint('   %s' % matched_pairs)
        print("   ID variables %s ")
        ppt.pprint('   %s' % IDcols)
        print("   Target variable %s ")
        ppt.pprint('   %s' % depVar)
    elif verbose==1 and len(preds) > max_cols_analyzed:
        print('   Total columns > %d, too numerous to list.' %max_cols_analyzed)
    features_dict = dict([('IDcols',IDcols),('cols_delete',cols_delete),('categorical_vars',categorical_vars), (
                        'lat_vars',lats),('lon_vars',lons),('matched_pairs',matched_pairs), ('int_vars',int_vars),
                        ('continuous_vars',continuous_vars),('discrete_string_vars',discrete_string_vars),
                        ('nlp_vars',nlp_vars), ('date_vars',date_vars)])
    return features_dict
#######################################################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        if verbose==1:
            print('Data Set columns info:')
            for col in data.columns:
                print('* %s: %d nulls, %d unique vals, most common: %s' % (
                        col,
                        data[col].isnull().sum(),
                        data[col].nunique(),
                        data[col].value_counts().head(2).to_dict()
                    ))
            print('--------------------------------------------------------------------')
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
from collections import defaultdict
def classify_columns(df_preds, model_options={}, verbose=0):
    """
    This actually does Exploratory data analysis - it means this function performs EDA
    ######################################################################################
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    ### if a variable has more than this many chars, it will be treated like a NLP variable

    max_nlp_char_size = check_model_options(model_options, "nlp_char_limit", 30)
    ### if a variable has more than this limit, it will not be treated like a cat variable #
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = check_model_options(model_options, "variable_cat_limit", 30)
    max_cols_to_print = 30
    #### Make this limit low so that float variables below this limit become cat vars ###
    float_limit = 15
    print('############## C L A S S I F Y I N G  V A R I A B L E S  ####################')
    print('Classifying variables in data set...')
    def add(a,b):
        return a+b
    sum_all_cols = defaultdict(list)
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                   ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete
    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    var_df['date_time'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    ###### This is where we detect categorical variables based on category limit #######
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and len(train[col].value_counts()
                        ) >= int(0.9*len(train)) and col not in string_bool_vars:
                try:
                    pd.to_datetime(train[col],infer_datetime_format=True)
                    var_df.loc[var_df['index']==col,'date_time'] = 1
                except:
                    var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                try:
                    pd.to_datetime(train[col],infer_datetime_format=True)
                    var_df.loc[var_df['index']==col,'date_time'] = 1
                except:
                    var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                try:
                    pd.to_datetime(train[col],infer_datetime_format=True)
                    var_df.loc[var_df['index']==col,'date_time'] = 1
                except:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars+date_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 1 if x['date_time']==1 else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    ###### In Tensorflow there is no need to create age variables from year-dates. Hence removing them!
    for date_var in copy_date_vars:
        if train[date_var].dtype in ['int16','int32','int64']:
            if train[date_var].min() >= 1900 or train[date_var].max() <= 2050:
                ### if it is between these numbers, its probably a year so avoid adding it
                date_items = train[date_var].dropna(axis=0).apply(str).apply(len).values
                if all(date_items[0] == item for item in date_items):
                    if date_items[0] == 4:
                        print('    Changing %s from date-var to int-var' %date_var)
                        int_vars.append(date_var)
                        date_vars.remove(date_var)
                        continue
        else:
            date_items = train[date_var].dropna(axis=0).apply(str).apply(len).values
            #### In some extreme cases, 4 digit date variables are not useful
            if all(date_items[0] == item for item in date_items):
                if date_items[0] == 4:
                    print('    Changing %s from date-var to discrete-string-var' %date_var)
                    discrete_string_vars.append(date_var)
                    date_vars.remove(date_var)
                    continue
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    ########  V E R Y    I M P O R T A N T   ###################################################
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    cols_delete = find_remove_duplicates(cols_delete+id_vars)
    sum_all_cols['cols_delete'] = cols_delete
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    cat_vars_copy = copy.deepcopy(cat_vars)
    for each_cat in cat_vars_copy:
        if len(train[each_cat].value_counts()) > cat_limit:
            discrete_string_vars.append(each_cat)
            cat_vars.remove(each_cat)
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    #########  The variables can now be printed ##############

    if verbose == 1:
        print("    Number of Numeric Columns = ", len(continuous_vars))
        print("    Number of Integer-Categorical Columns = ", len(int_vars))
        print("    Number of String-Categorical Columns = ", len(cat_vars))
        print("    Number of Factor-Categorical Columns = ", len(factor_vars))
        print("    Number of String-Boolean Columns = ", len(string_bool_vars))
        print("    Number of Numeric-Boolean Columns = ", len(num_bool_vars))
        print("    Number of Discrete String Columns = ", len(discrete_string_vars))
        print("    Number of NLP String Columns = ", len(nlp_vars))
        print("    Number of Date Time Columns = ", len(date_vars))
        print("    Number of ID Columns = ", len(id_vars))
        print("    Number of Columns to Delete = ", len(cols_delete))
    if verbose == 2:
        marthas_columns(df_preds,verbose=1)
    if verbose >=1 and orig_cols_total > max_cols_to_print:
        print("    Numeric Columns: %s" %continuous_vars[:max_cols_to_print])
        print("    Integer-Categorical Columns: %s" %int_vars[:max_cols_to_print])
        print("    String-Categorical Columns: %s" %cat_vars[:max_cols_to_print])
        print("    Factor-Categorical Columns: %s" %factor_vars[:max_cols_to_print])
        print("    String-Boolean Columns: %s" %string_bool_vars[:max_cols_to_print])
        print("    Numeric-Boolean Columns: %s" %num_bool_vars[:max_cols_to_print])
        print("    Discrete String Columns: %s" %discrete_string_vars[:max_cols_to_print])
        print("    NLP text Columns: %s" %nlp_vars[:max_cols_to_print])
        print("    Date Time Columns: %s" %date_vars[:max_cols_to_print])
        print("    ID Columns: %s" %id_vars[:max_cols_to_print])
        print("    Columns that will not be considered in modeling: %s" %cols_delete[:max_cols_to_print])
    ##### now collect all the column types and column names into a single dictionary to return!
    #### Since cols_delete and id_vars have the same columns, you need to subtract id_vars from this!
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()]) - len(id_vars)
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        #print('        This does not include the Target column(s)')
    else:
        print('Number columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
#################################################################################
from collections import defaultdict
def nested_dictionary():
    return defaultdict(nested_dictionary)
############################################################################################
def check_model_options(model_options, name, default):
    try:
        if model_options[name]:
            value = model_options[name]
        else:
            value = default
    except:
        value = default
    return value
#####################################################################################
def classify_features_using_pandas(data_sample, target, model_options={}, verbose=0):
    """
    If you send in a small pandas dataframe with the name of target variable(s), you will get back
    all the features classified by type such as dates, cats, ints, floats and nlps. This is all done using pandas.
    """
    ######   This is where you get the cat_vocab_dict is created in the form of feats_max_min #####
    feats_max_min = nested_dictionary()
    print_features = False
    nlps = []
    bools = []
    ### if a variable has more than this many chars, it will be treated like a NLP variable
    nlp_char_limit = check_model_options(model_options, "nlp_char_limit", 30)
    ### if a variable has more than this limit, it will not be treated like a cat variable #
    cat_limit = check_model_options(model_options, "variable_cat_limit", 30)
    ### Classify features using the previously define function #############
    var_df1 = classify_features(data_sample, target, model_options, verbose=verbose)
    #####  This might be useful for users to know whether to use feature-crosses or not ###
    stri, numi, cat_feature_cross_flag = fast_classify_features(data_sample)
    convert_cols = []
    if len(numi['veryhighcats']) > 0:
        convert_cols =  numi['veryhighcats']
    if convert_cols:
        var_df1['int_vars'] = left_subtract(var_df1['int_vars'], convert_cols)
        var_df1['continuous_vars'] = var_df1['continuous_vars'] + convert_cols

    dates = var_df1['date_vars']
    cats = var_df1['categorical_vars']
    discrete_strings = var_df1['discrete_string_vars']
    lats = var_df1['lat_vars']
    lons = var_df1['lon_vars']
    ignore_variables = var_df1['cols_delete']
    all_ints = var_df1['int_vars']
    if isinstance(target, list):
        preds = [x for x in  list(data_sample) if x not in target+ignore_variables]
        feats_max_min['predictors_in_train'] = [x for x in  list(data_sample) if x not in target]
    else:
        preds = [x for x in  list(data_sample) if x not in [target]+ignore_variables]
        feats_max_min['predictors_in_train'] = [x for x in  list(data_sample) if x not in [target]]
    #### Take(1) always displays only one batch only if num_epochs is set to 1 or a number. Otherwise No print! ########
    #### If you execute the below code without take, then it will go into an infinite loop if num_epochs was set to None.
    if verbose >= 1 and target:
            print(f"printing first five values of {target}: {data_sample[target].values[:5]}")
    if len(preds) <= 30:
        print_features = True
    if print_features and verbose > 1:
        print("printing features and their max, min, datatypes in one batch ")
    ###### Now we do the creation of cat_vocab_dict though it is called feats_max_min here #####
    floats = []
    for key in preds:
        if data_sample[key].dtype in ['object'] or str(data_sample[key].dtype) == 'category':
            feats_max_min[key]["dtype"] = "string"
        elif data_sample[key].dtype in ['bool']:
            feats_max_min[key]["dtype"] = "bool"
            bools.append(key)
            if key in cats:
                cats.remove(key)
        elif str(data_sample[key].dtype).split("[")[0] in ['datetime64','datetime32','datetime16']:
            feats_max_min[key]["dtype"] = "string"
        elif data_sample[key].dtype in [np.int16, np.int32, np.int64]:
            if key in convert_cols:
                feats_max_min[key]["dtype"] = np.float32
                floats.append(key)
            else:
                feats_max_min[key]["dtype"] = np.int32
        else:
            floats.append(key)
            feats_max_min[key]["dtype"] = np.float32
        if feats_max_min[key]['dtype'] in [np.int16, np.int32, np.int64,
                                np.float16, np.float32, np.float64]:
            ##### This is for integer and float variables #######
            if key in lats+lons:
                ### For lats and lons you need the vocab to create bins using pd.qcut ####
                vocab = data_sample[key].unique()
                feats_max_min[key]["vocab"] = vocab
                feats_max_min[key]['size_of_vocab'] = len(vocab)
                feats_max_min[key]["max"] = max(data_sample[key].values)
                feats_max_min[key]["min"] = min(data_sample[key].values)
            else:
                if feats_max_min[key]['dtype'] in [np.int16, np.int32, np.int64]:
                    vocab = data_sample[key].unique()
                    feats_max_min[key]["vocab"] = vocab
                    feats_max_min[key]['size_of_vocab'] = len(vocab)
                else:
                    ### For the rest of the numeric variables, you just need mean and variance ###
                    vocab = data_sample[key].unique()
                    feats_max_min[key]["vocab_min_var"] = [data_sample[key].mean(), data_sample[key].var()]
                    feats_max_min[key]["vocab"] = vocab
                    feats_max_min[key]['size_of_vocab'] = len(vocab)
            feats_max_min[key]["max"] = max(data_sample[key].values)
            feats_max_min[key]["min"] = min(data_sample[key].values)
        elif feats_max_min[key]['dtype'] in ['bool']:
            ### we are going to convert boolean to float type #####
            vocab = data_sample[key].unique()
            full_array = data_sample[key].values
            full_array = np.array([0.0 if type(x) == float else float(x) for x in full_array])
            ### Don't change the next line even though it appears wrong. I have tested and it works!
            vocab = [0.0 if type(x) == float else float(x) for x in vocab]
            feats_max_min[key]["vocab_min_var"] = [full_array.mean(), full_array.var()]
            feats_max_min[key]["vocab"] = vocab
            feats_max_min[key]['size_of_vocab'] = len(vocab)
        elif feats_max_min[key]['dtype'] in ['string']:
            if np.mean(data_sample[key].fillna("missing").map(len)) >= nlp_char_limit:
                ### This is for NLP variables. You want to remove duplicates #####
                if key in dates:
                    continue
                elif key in cats:
                    cats.remove(key)
                    var_df1['categorical_vars'] = cats
                elif key in discrete_strings:
                    discrete_strings.remove(key)
                    var_df1['discrete_string_vars'] = discrete_strings
                print('    %s is detected as an NLP variable' %key)
                if key not in var_df1['nlp_vars']:
                    var_df1['nlp_vars'].append(key)
                feats_max_min[key]['seq_length'] = int(data_sample[key].fillna("missing").map(len).max())
                num_words_in_each_row = data_sample[key].fillna("missing").map(lambda x: len(x.split(" "))).mean()
                num_rows_in_data = model_options['DS_LEN']
                feats_max_min[key]['size_of_vocab'] = int(num_rows_in_data*num_words_in_each_row)
            else:
                ### This is for string variables ########
                ####  Now we select features if they are present in the data set ###
                #feats_max_min[key]["vocab"] = data_sample[key].unique()
                vocab = data_sample[key].unique()
                vocab = ['missing' if type(x) != str  else x for x in vocab]
                feats_max_min[key]["vocab"] = vocab
                feats_max_min[key]['size_of_vocab'] = len(vocab)
                #feats_max_min[key]['size_of_vocab'] = len(feats_max_min[key]["vocab"])
        else:
            ####  Now we treat bool and other variable types ###
            #feats_max_min[key]["vocab"] = data_sample[key].unique()
            vocab = data_sample[key].unique()
            #### just leave this as is - it works for other data types
            vocab = ['missing' if type(x) == str  else x for x in vocab]
            feats_max_min[key]["vocab"] = vocab
            feats_max_min[key]['size_of_vocab'] = len(vocab)
            #feats_max_min[key]['size_of_vocab'] = len(feats_max_min[key]["vocab"])
        if print_features and verbose > 1:
            print("  {!r:20s}: {}".format(key, data_sample[key].values[:4]))
            print("  {!r:25s}: {}".format('    size of vocab', feats_max_min[key]["size_of_vocab"]))
            print("  {!r:25s}: {}".format('    max', feats_max_min[key]["max"]))
            print("  {!r:25s}: {}".format('    min', feats_max_min[key]["min"]))
            print("  {!r:25s}: {}".format('    dtype', feats_max_min[key]["dtype"]))
    if not print_features:
        print('Number of variables in dataset is too numerous to print...skipping print')
    ##### Make some changes to integer variables to select those with less than certain category limits ##
    ints = [ x for x in all_ints if feats_max_min[x]['size_of_vocab'] > cat_limit and x not in floats]

    int_bools = [ x for x in all_ints if feats_max_min[x]['size_of_vocab'] == 2 and x not in floats]

    int_cats = [ x for x in all_ints if feats_max_min[x]['size_of_vocab'] <= cat_limit and x not in floats+int_bools]

    var_df1['int_vars'] = ints
    var_df1['int_cats'] = int_cats
    var_df1['int_bools'] = int_bools
    var_df1["continuous_vars"] = floats
    var_df1['bools'] = bools
    #### It is better to have a baseline number for the size of the dataset here ########
    feats_max_min['DS_LEN'] = len(data_sample)
    ### check if cat_vocab_dict has cat_feature_cross_flag in it ###
    if "cat_feature_cross_flag" in model_options.keys():
        ### Since they have asked to do cat feature crossing, then do it ####
        model_options["cat_feature_cross_flag"] = cat_feature_cross_flag
        print('performing feature crossing for %s variables' %cat_feature_cross_flag)
    else:
        ### If there is no input for cat_feature_cross_flag, then don't do it ###
        cat_feature_cross_flag = ""
        print('Not performing feature crossing for categorical nor integer variables' )
    return var_df1, feats_max_min
############################################################################################
def EDA_classify_and_return_cols_by_type(df1, nlp_char_limit=20):
    """
    EDA stands for Exploratory data analysis. This function performs EDA - hence the name
    ########################################################################################
    This handy function classifies your columns into different types : make sure you send only predictors.
    Beware sending target column into the dataframe. You don't want to start modifying it.
    #####################################################################################
    It returns a list of categorical columns, integer cols and float columns in that order.
    """
    ### Let's find all the categorical excluding integer columns in dataset: unfortunately not all integers are categorical!
    catcols = df1.select_dtypes(include='object').columns.tolist() + df1.select_dtypes(include='category').columns.tolist()
    cats = copy.deepcopy(catcols)
    nlpcols = []
    for each_cat in cats:
        try:
            if df1[each_cat].map(len).mean() >=nlp_char_limit:
                nlpcols.append(each_cat)
                catcols.remove(each_cat)
        except:
            continue
    intcols = df1.select_dtypes(include='integer').columns.tolist()
    int_cats = [ x for x in intcols if df1[x].nunique() <= 30 and x not in idcols]
    intcols = left_subtract(intcols, int_cats)
    # let's find all the float numeric columns in data
    floatcols = df1.select_dtypes(include='float').columns.tolist()
    return catcols, int_cats, intcols, floatcols, nlpcols
############################################################################################
def EDA_classify_features(train, target, idcols, nlp_char_limit=20):
    ### Test Labeler is a very important dictionary that will help transform test data same as train ####
    test_labeler = defaultdict(list)

    #### all columns are features except the target column and the folds column ###
    if isinstance(target, str):
        features = [x for x in list(train) if x not in [target]+idcols]
    else:
        ### in this case target is a list and hence can be added to idcols
        features = [x for x in list(train) if x not in target+idcols]

    ### first find all the types of columns in your data set ####
    cats, int_cats, ints, floats, nlps = EDA_classify_and_return_cols_by_type(train[features],
                                                                            nlp_char_limit)

    numeric_features = ints + floats
    categoricals_features = copy.deepcopy(cats)
    nlp_features = copy.deepcopy(nlps)

    test_labeler['categoricals_features'] = categoricals_features
    test_labeler['numeric_features'] = numeric_features
    test_labeler['nlp_features'] = nlp_features

    return cats, int_cats, ints, floats, nlps
#############################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst

#############################################################################################
def find_number_bins(series):
    """
    Input must be a pandas series. Otherwise it will blow up. Be careful!
    Returns the recommended number of bins for any Series in pandas
    Input must be a float or integer column. Don't send in alphabetical series!
    """
    return int(np.log2(series.nunique())+1)
#########################################################################################
import re
def find_words_in_list(words, in_list):
    result = []
    for each_word in words:
        for in_src in in_list:
            if re.findall(each_word, in_src):
                result.append(in_src)
    return list(set(result))

#############################################################################################
from collections import defaultdict
def find_latitude_longitude_columns_in_df(df, verbose=0):
    matched_pairs = []
    lats, lat_keywords = find_latitude_columns(df, verbose)
    lons, lon_keywords = find_longitude_columns(df, verbose)
    if len(lats) > 0 and len(lons) > 0:
        if len(lats) >= 1:
            for each_lat in lats:
                for each_lon in lons:
                    if lat_keywords[each_lat] == lon_keywords[each_lon]:
                        matched_pairs.append((each_lat, each_lon))
    if len(matched_pairs) > 0 and verbose:
        print('Matched pairs of latitudes and longitudes: %s' %matched_pairs)
    return lats, lons, matched_pairs

def find_latitude_columns(df, verbose=0):
    columns = df.select_dtypes(include='float').columns.tolist() + df.select_dtypes(include='object').columns.tolist()
    lat_words = find_words_in_list(['Lat','lat','LAT','Latitude','latitude','LATITUDE'], columns)
    sel_columns = lat_words[:]
    lat_keywords = defaultdict(str)
    if len(columns) > 0:
        for lat_word in columns:
            lati_keyword = find_latitude_keyword(lat_word, columns, sel_columns)
            if not lati_keyword == '':
                lat_keywords[lat_word] = lat_word.replace(lati_keyword,'')
    ###### This is where we find whether they are truly latitudes ############
    print('    possible latitude columns in dataset: %s' %sel_columns)
    sel_columns_copy = copy.deepcopy(sel_columns)
    for sel_col in sel_columns_copy:
        if not lat_keywords[sel_col]:
            sel_columns.remove(sel_col)
    if len(sel_columns) == 0:
        print('        after further analysis, no latitude columns found')
    else:
        print('        after further analysis, selected latitude columns = %s' %sel_columns)
     #### If there are any more columns left, then do further analysis #######
    if len(sel_columns) > 0:
        sel_cols_float = df[sel_columns].select_dtypes(include='float').columns.tolist()
        if len(sel_cols_float) > 0:
            for sel_column in sel_cols_float:
                if df[sel_column].isnull().sum() > 0:
                    print('Null values in possible latitude column %s. Removing it' %sel_column)
                    sel_columns.remove(sel_column)
                    continue
                if df[sel_column].min() >= -90 and df[sel_column].max() <= 90:
                    if verbose:
                        print('        %s found as latitude column' %sel_column)
                    if sel_column not in sel_columns:
                        sel_columns.append(sel_column)
                        lati_keyword = find_latitude_keyword(sel_column, columns, sel_columns)
                        if not lati_keyword == '':
                            lat_keywords[lat_word] = sel_column.replace(lati_keyword,'')
                else:
                    sel_columns.remove(sel_column)
        sel_cols_string = df[sel_columns].select_dtypes(include='object').columns.tolist()
        if len(sel_cols_string) > 0:
            for sel_column in sel_cols_string:
                if len(df[df[sel_column].str.endswith(('N','S'))]) > 0:
                    if verbose:
                        print('        %s found as latitude column' %sel_column)
                    if sel_column not in sel_columns:
                        sel_columns.append(sel_column)
                        lati_keyword = find_latitude_keyword(sel_column, columns, sel_columns)
                        if not lati_keyword == '':
                            lat_keywords[lat_word] = sel_column.replace(lati_keyword,'')
                else:
                    sel_columns.remove(sel_column)
    return sel_columns, lat_keywords

def find_latitude_keyword(lat_word, columns, sel_columns=[]):
    lat_keywords = defaultdict(str)
    ####  This is where we find the text that is present in column related to latitude ##
    if len(columns) > 0:
        if lat_word.lower() == 'lat':
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'lat'
        elif lat_word.lower() == 'latitude':
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'latitude'
        elif 'latitude' in lat_word.lower().split(" "):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'latitude'
        elif 'latitude' in lat_word.lower().split("_"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'latitude'
        elif 'latitude' in lat_word.lower().split("-"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'latitude'
        elif 'latitude' in lat_word.lower().split("/"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'latitude'
        elif 'lat' in lat_word.lower().split(" "):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'lat'
        elif 'lat' in lat_word.lower().split("_"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'lat'
        elif 'lat' in lat_word.lower().split("-"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'lat'
        elif 'lat' in lat_word.lower().split("/"):
            if lat_word not in sel_columns:
                sel_columns.append(lat_word)
            lat_keywords[lat_word] = 'lat'
    return lat_keywords[lat_word]

def find_longitude_keyword(lon_word, columns, sel_columns=[]):
    lon_keywords = defaultdict(str)
    ####  This is where we find the text that is present in column related to longitude ##
    if len(columns) > 0:
        if lon_word.lower() == 'lon':
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'lon'
        elif lon_word.lower() == 'longitude':
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'longitude'
        elif 'longitude' in lon_word.lower().split(" "):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'longitude'
        elif 'longitude' in lon_word.lower().split("_"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'longitude'
        elif 'longitude' in lon_word.lower().split("-"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'longitude'
        elif 'longitude' in lon_word.lower().split("/"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'longitude'
        elif 'lon' in lon_word.lower().split(" "):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'lon'
        elif 'lon' in lon_word.lower().split("_"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'lon'
        elif 'lon' in lon_word.lower().split("-"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'lon'
        elif 'lon' in lon_word.lower().split("/"):
            if lon_word not in sel_columns:
                sel_columns.append(lon_word)
            lon_keywords[lon_word] = 'lon'
    return lon_keywords[lon_word]

def find_longitude_columns(df, verbose=0):
    columns = df.select_dtypes(include='float').columns.tolist() + df.select_dtypes(include='object').columns.tolist()
    lon_words = find_words_in_list(['Lon','lon','LON','Longitude','Longitude', "LONGITUDE"], columns)
    sel_columns = lon_words[:]
    lon_keywords = defaultdict(str)
    if len(columns) > 0:
        for lon_word in columns:
            long_keyword = find_longitude_keyword(lon_word, columns, sel_columns)
            if not long_keyword == '':
                lon_keywords[lon_word] = lon_word.replace(long_keyword,'')
    #####  This is where we test whether they are indeed longitude columns ####
    print('    possible longitude columns in dataset: %s' %sel_columns)
    sel_columns_copy = copy.deepcopy(sel_columns)
    for sel_col in sel_columns_copy:
        if not lon_keywords[sel_col]:
            sel_columns.remove(sel_col)
    if len(sel_columns) == 0:
        print('        after further analysis, no longitude columns found')
    else:
        print('        after further analysis, selected longitude columns = %s' %sel_columns)
    ###### This is where we find whether they are truly longitudes ############
    if len(sel_columns) > 0:
        sel_cols_float = df[sel_columns].select_dtypes(include='float').columns.tolist()
        if len(sel_cols_float) > 0:
            for sel_column in sel_cols_float:
                if df[sel_column].isnull().sum() > 0:
                    print('Null values in possible longitude column %s. Removing it' %sel_column)
                    sel_columns.remove(sel_column)
                    continue
                if df[sel_column].min() >= -180 and df[sel_column].max() <= 180:
                    if verbose:
                        print('        %s found as longitude column' %sel_column)
                    if sel_column not in sel_columns:
                        sel_columns.append(sel_column)
                        long_keyword = find_longitude_keyword(sel_column, columns, sel_columns)
                        if not long_keyword == '':
                            lon_keywords[lon_word] = sel_column.replace(long_keyword,'')
                else:
                    sel_columns.remove(sel_column)
        sel_cols_string = df[sel_columns].select_dtypes(include='object').columns.tolist()
        if len(sel_cols_string) > 0:
            for sel_column in sel_cols_string:
                if len(df[df[sel_column].str.endswith(('N','S'))]) > 0:
                    if verbose:
                        print('        %s found as longitude column' %sel_column)
                    if sel_column not in sel_columns:
                        sel_columns.append(sel_column)
                        long_keyword = find_longitude_keyword(sel_column, columns, sel_columns)
                        if not long_keyword == '':
                            lon_keywords[lon_word] = sel_column.replace(long_keyword,'')
                else:
                    sel_columns.remove(sel_column)
    return sel_columns, lon_keywords
###########################################################################################
from collections import defaultdict
def nested_dictionary():
    return defaultdict(nested_dictionary)
############################################################################################
def classify_dtypes_using_TF2(data_sample, idcols, verbose=0):
    """
    If you send in a batch of Ttf.data.dataset with the name of target variable(s), you will get back
    all the features classified by type such as cats, ints, floats and nlps. This is all done using TF2.
    """
    print_features = False
    nlps = []
    nlp_char_limit = 30
    all_ints = []
    floats = []
    cats = []
    int_vocab = 0
    feats_max_min = nested_dictionary()

    #### Take(1) always displays only one batch only if num_epochs is set to 1 or a number. Otherwise No print! ########
    #### If you execute the below code without take, then it will go into an infinite loop if num_epochs was set to None.
    if data_sample.element_spec[0][preds[0]].shape[0] is None:
        for feature_batch, label_batch in data_sample.take(1):
            if verbose >= 1:
                print(f"{target}: {label_batch[:4]}")
            if len(feature_batch.keys()) <= 30:
                print_features = True
                if verbose >= 1:
                    print("features and their max, min, datatypes in one batch of size: ",batch_size)
                for key, value in feature_batch.items():
                    feats_max_min[key]["dtype"] = data_sample.element_spec[0][key].dtype
                    if feats_max_min[key]['dtype'] in [tf.float16, tf.float32, tf.float64]:
                        ## no need to find vocab of floating point variables!
                        floats.append(key)
                    elif feats_max_min[key]['dtype'] in [tf.int16, tf.int32, tf.int64]:
                        ### if it is an integer var, it is worth finding their vocab!
                        all_ints.append(key)
                        int_vocab = tf.unique(value)[0].numpy().tolist()
                        feats_max_min[key]['size_of_vocab'] = len(int_vocab)
                    elif feats_max_min[key]['dtype'] in [tf.string]:
                        if tf.reduce_mean(tf.strings.length(feature_batch[key])).numpy() >= nlp_char_limit:
                            print('%s is detected as an NLP variable')
                            nlps.append(key)
                        else:
                            cats.append(key)
            if not print_features:
                print('Number of variables in dataset is too numerous to print...skipping print')

    ints = [ x for x in all_ints if feats_max_min[x]['size_of_vocab'] > 30 and x not in idcols]

    int_cats = [ x for x in all_ints if feats_max_min[x]['size_of_vocab'] <= 30 and x not in idcols]

    return cats, int_cats, ints, floats, nlps
############################################################################################
# Define feature columns(Including feature engineered ones )
# These are the features which come from the TF Data pipeline
def create_feature_cols(data_batches, preds):
    #Keras format features
    keras_dict_input = {}
    if data_batches.element_spec[0][preds[0]].shape[0] is None:
        print("Creating keras features dictionary...")
        for feature_batch, label_batch in data_batches.take(1):
            for key, _ in feature_batch.items():
                k_month = tf.keras.Input(name=key, shape=(1,), dtype=tf.string)
                keras_dict_input[key] = k_month
    print('    completed.')
    return({'K' : keras_dict_input})
##############################################################################################
# Removes duplicates from a list to return unique values - USED ONLYONCE
def find_remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
#################################################################################
from collections import defaultdict
import copy
def fast_classify_features(df):
    """
    This is a very fast way to get a handle on what a dataset looks like. Just send in df and get a print.
    Nothing is returned. You just get a printed number of how many types of features you have in dataframe.
    """
    num_list = df.select_dtypes(include='integer').columns.tolist()
    float_list = df.select_dtypes(include='float').columns.tolist()
    str_list = left_subtract(df.columns.tolist(), num_list+float_list)
    all_list = [str_list, num_list]
    str_dict = defaultdict(dict)
    int_dict = defaultdict(dict)
    for inum, dicti in enumerate([str_dict, int_dict]):
        bincols = []
        catcols = []
        highcols = []
        numcols = []
        for col in all_list[inum]:
            leng = len(df[col].value_counts())
            if leng <= 2:
                bincols.append(col)
            elif leng > 2 and leng <= 15:
                catcols.append(col)
            elif leng >15 and leng <100:
                highcols.append(col)
            else:
                numcols.append(col)
        dicti['bincols'] = bincols
        dicti['catcols'] = catcols
        dicti['highcats'] = highcols
        dicti['veryhighcats'] = numcols
        if inum == 0:
            str_dict = copy.deepcopy(dicti)
            print('Distribution of string columns in datatset:')
            print('    number of binary = %d, cats = %d, high cats = %d, very high cats = %d' %(
                len(bincols), len(catcols), len(highcols), len(numcols)))
        else:
            print('Distribution of integer columns in datatset:')
            int_dict = copy.deepcopy(dicti)
            print('    number of binary = %d, cats = %d, high cats = %d, very high cats = %d' %(
                len(bincols), len(catcols), len(highcols), len(numcols)))
    ###   Check if worth doing cat_feature_cross_flag on this dataset ###
    int_dict['floats'] = float_list
    print('Distribution of floats: floats = %d' %len(float_list))
    print('Data Transformation Advisory:')
    cat_feature_cross_flag = []
    if len(str_dict['bincols']+str_dict['catcols']) > 2 and len(str_dict['bincols']+str_dict['catcols']) <= 10:
        cat_feature_cross_flag.append("cat")
    if len(int_dict['bincols']+int_dict['catcols']) > 2 and len(int_dict['bincols']+int_dict['catcols']) <= 10:
        cat_feature_cross_flag.append("num")
    if cat_feature_cross_flag:
        if "cat" in cat_feature_cross_flag:
            cat_feature_cross_flag = "cat"
            print('    performing categorical feature crosses: changed cat_feat_cross_flag to "cat"')
        elif "num" in cat_feature_cross_flag:
            cat_feature_cross_flag = "num"
            print('    performing integer feature crosses: changed cat_feat_cross_flag to "num" ')
        elif "cat" in cat_feature_cross_flag and "num" in cat_feature_cross_flag:
            cat_feature_cross_flag = "both"
            print('    performing both integer and cat feature crosses: changed cat_feat_cross_flag to "both" ')
    else:
        cat_feature_cross_flag = ""
    if len(int_dict['veryhighcats']) > 0:
        print('    transformed %s from integer to float' %int_dict['veryhighcats'])
    return str_dict, int_dict, cat_feature_cross_flag
###################################################################################################
