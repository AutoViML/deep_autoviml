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
#     deep_auto_viml - build and test multiple Tensorflow 2.0 models and pipelines
#     Python v3.6+ tensorflow v2.4.1+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# data pipelines and feature engg here

# pre-defined TF2 Keras models and your own models here 

# Utils
from .classify_features import classify_features_using_pandas
from .classify_features import check_model_options
# Utils
from deep_autoviml.utilities.utilities import print_one_row_from_tf_dataset, print_one_row_from_tf_label
from deep_autoviml.utilities.utilities import My_LabelEncoder, print_one_image_from_dataset

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
# TensorFlow ≥2.4 is required
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from tensorflow.keras import layers
from tensorflow import keras
############################################################################################
#### probably the most handy function of all!
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
import re
def find_words_in_list(words, in_list):
    result = []
    for each_word in words:
        for in_src in in_list:
            if re.findall(each_word, in_src):
                result.append(in_src)
    return list(set(result))

##############################################################################################
def find_problem_type(train, target, model_options={}, verbose=0) :
    """
    ############################################################################
    #####   Now find the problem type of this train dataset using its target variable
    ############################################################################
    """
    target = copy.deepcopy(target)
    ### this determines the number of categories to name integers as classification ##
    ### if a variable has more than this limit, it will not be treated like a cat variable #
    cat_limit = check_model_options(model_options, "variable_cat_limit", 30)
    float_limit = 15 ### this limits the number of float variable categories for it to become cat var
    model_label = 'Single_Label'
    model_class = 'Classification'
    if isinstance(target, str):
        if target == '':
            model_class ='Clustering'
            model_label = 'Single_Label'
            return model_class, model_label,  target
        targ = copy.deepcopy(target)
        target = [target]
    elif isinstance(target, list):
        if len(target) == 1:
            targ = target[0]
        else:
            targ = target[0]
            model_label = 'Multi_Label'
    else:
        print('target is Not detected. Default chosen is %s, %s' %(model_class, model_label))
    ####  This is where you detect what kind of problem it is #################

    if  train[targ].dtype in ['int64', 'int32','int16']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= cat_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    elif  train[targ].dtype in ['float']:
        if len(train[targ].unique()) <= 2:
            model_class = 'Classification'
        elif len(train[targ].unique()) > 2 and len(train[targ].unique()) <= float_limit:
            model_class = 'Multi_Classification'
        else:
            model_class = 'Regression'
    else:
        if len(train[targ].unique()) <= 2:
            model_class = 'Classification'
        else:
            model_class = 'Multi_Classification'
    ########### print this for the start of next step ###########
    print('    Model type is %s and %s problem' %(model_class,model_label))
    return model_class, model_label,  target
######################################################################################

def transform_train_target(train_target, target, modeltype, model_label, cat_vocab_dict):
    train_target = copy.deepcopy(train_target)
    cat_vocab_dict = copy.deepcopy(cat_vocab_dict)
    ### Just have to change the target from string to Numeric in entire dataframe! ###
    
    if modeltype != 'Regression':
        if model_label == 'Multi_Label':
            target_copy = copy.deepcopy(target)
            print('Train target shape = %s' %(train_target.shape,))
            #### This is for multi-label problems #####
            cat_vocab_dict['target_le'] = []
            for each_target in target_copy:
                cat_vocab_dict[each_target+'_original_classes'] = np.unique(train_target[target])
                target_le = My_LabelEncoder()
                print('Transforming %s target labels...' %each_target)
                print('    Original target labels data type is %s ' %train_target[each_target].dtype)
                train_values = target_le.fit_transform(train_target[each_target])
                cat_vocab_dict[each_target+'_transformed_classes'] = np.unique(train_values)
                train_target[each_target] = train_values
                cat_vocab_dict['target_le'].append(target_le)
                print('%s transformed as follows: %s' %(each_target, target_le.transformer))
                print('    transformed target labels data type to numeric or ordered from 0')
        else:
            #### This is for Single Label problems ####
            cat_vocab_dict['original_classes'] = np.unique(train_target[target])
            target_le = My_LabelEncoder()
            print('Transforming %s target labels...' %target)
            print('    Original labels dtype is %s ' %train_target[target].dtype)
            train_values = target_le.fit_transform(train_target[target])
            cat_vocab_dict['transformed_classes'] = np.unique(train_values)
            train_target[target] = train_values
            cat_vocab_dict['target_le'] = target_le
            print('%s transformed as follows: %s' %(target, target_le.transformer))
            print('    transformed target labels data type to numeric or ordered from 0')
    else:
        target_le = ""
        cat_vocab_dict['target_le'] = target_le
        print('No Target transformation needed since target dtype is numeric')
    train_target = train_target[target]
    return train_target, cat_vocab_dict

def split_combined_ds_into_two(x, usecols, preds):
    """
    This is useful for splitting a single dataset which has both features and labels into two.
    usecols is basically target column in the form of a list: [target]
    preds is basically predictor columns in the form of a list: a list of predictors
    """
    labels = {k: x[k] for k in x if k in usecols}
    features = {k: x[k] for k in x if k in preds}
    return (features, labels)
######################################################################################################
import pathlib
import os
import random
def load_train_data_file(train_datafile, target, keras_options, model_options, verbose=0):
    """
    This handy function loads a file from a local or remote machine provided the filename and path are given.
    It loads the file(s) into a Tensorflow Dataset using the make_csv_dataset function from Tensorflow 2.0
    """
    http_url = False
    if find_words_in_list(['http'], [train_datafile]):
        print('http urls file: will be loaded into pandas and then into tensorflow datasets')
        http_url = True
    try:
        DS_LEN = model_options['DS_LEN']
    except:
        ### Choose a default option in case it is not given
        DS_LEN = 10000
    shuffle_flag = False
    #### Set some default values in case model options is not set ##
    try:
        sep = model_options["sep"]
    except:
        sep = ","
    try:
        header = model_options["header"]
    except:
        header = 0
    try:
        csv_encoding = model_options["csv_encoding"]
    except:
        csv_encoding = 'utf-8'
    #################################################################################
    try:
        compression = None
        ### see if there is a . in the file name. If it is, then do this process.
        split_str = train_datafile.split(".")[-1]
        if split_str=='csv':
            print("CSV file being loaded into tf.data.Dataset")
            compression_type = None
        elif split_str=='zip' or split_str=='gz':
            print("Zip file being loaded into tf.data.Dataset")
            compression_type="GZIP"
            compression = "gzip"
            print('    Using %s compression_type in make_csv_dataset argument' %compression_type)
        else:
            compression_type = None
    except:
        #### if . is not there, it means it is a folder and we need to collect all files in that folder
        font_csvs =  sorted(str(p) for p in pathlib.Path(train_datafile).glob("*.csv"))
        print('Printing the first 5 files in the %s folder:\n%s' %(train_datafile,font_csv[:5]))
        train_datafile_list = pathlib.Path(train_datafile).glob("*.csv")
        print('    collecting files matching this file pattern in directory: %s' %train_datafile_list)
        try:
            list_files = []
            filetype = train_datafile.split(".")[-1]
            list_files = [x for x in os.listdir(inpath) if x.endswith(filetype)]
            if list_files == []:
                print('No csv, tsv or Excel files found in the given directory')
                return
            else:
                print('%d files found in directory matching pattern: %s' %(len(list_files), train_datafile))
            ### now you must use this file_pattern in make_csv_dataset argument
            train_datafile = list_files[0]
        except:
            print('not able to collect files matching given pattern = %s' %train_datafile)
            return
    #################################################################################
    #### About 25% of the data or 10,000 rows which ever is higher is loaded #######
    if http_url:
        maxrows = 100000 ### set it very high so that all rows are read into dataframe ###
    else:
        maxrows = max(10000, int(0.25*DS_LEN))
    ### first load a small sample of the dataframe and the entire target if it needs transform
    if DS_LEN > 2*maxrows:
        print('Since number of rows > maxrows, loading a random sample of %d rows into pandas for EDA' %maxrows)
        ### we randomly sample every 2nd row until we get 10000
        skip_function = lambda x: (x != 0) and x % 5 and x < 2*maxrows
        ### load a small sample of data into a pandas dataframe ##
        train_small = pd.read_csv(train_datafile, sep=sep, skiprows=skip_function, header=header,
                        encoding=csv_encoding, nrows=maxrows, compression=compression)
    else:
        print('Since number of rows in file <= %d maxrows, loading entire file into pandas for EDA' %maxrows)
        ### load a small sample of data into a pandas dataframe ##
        train_small = pd.read_csv(train_datafile, sep=sep, nrows=maxrows, compression=compression,
                                header=header, encoding=csv_encoding) 
        ### this reads the entire file
    ##### Now detect modeltype if it is not given ###############
    try:
        modeltype = model_options["modeltype"]
    except:
        modeltype, model_label, usecols = find_problem_type(train_small, target, model_options, verbose)
    #### All column names in Tensorflow should have no spaces ! So you must convert them here!
    if header is None:
        sel_preds = ["col_"+str(x) for x in range(train_small.shape[1])]
    else:
        sel_preds = ["_".join(x.split(" ")) for x in list(train_small) ]
    if isinstance(target, str):
        target = "_".join(target.split(" "))
        model_label = 'Single_Label'
    else:
        target = ["_".join(x.split(" ")) for x in target ]
        model_label = 'Multi_Label'

    print('    Modified column names to fit no-spaces-in-column-names Rule in Tensorflow!')
    train_small.columns = sel_preds
    ### modeltype and usecols are very important to know before doing any processing #####    
    #### usecols is a very handy tool to handle a target which can be single label or multi-label!
    if modeltype == '':
        ### usecols is basically target in a list format. Very handy to know when target is a list.
        modeltype, _, usecols = find_problem_type(train_small, target, model_options, verbose)
    else:
        ### if modeltype is given, then do not find the model type using this function
        _,  _, usecols = find_problem_type(train_small, target, model_options, verbose)
    
    label_encode_flag = False
    ##########  Find small details about the data to help create the right model ###
    if modeltype == 'Classification' or modeltype == 'Multi_Classification':
        if isinstance(target, str):
            #### This is for Single-Label problems ########
            if train_small[target].dtype == 'object' or str(train_small[target].dtype).lower() == 'category':
                label_encode_flag = True
            elif 0 not in np.unique(train_small[target]):
                label_encode_flag = True ### label encoding must be done since no zero class!
            target_vocab = train_small[target].unique()
            num_classes = len(target_vocab)
        elif isinstance(target, list):
            #### This is for Multi-Label problems ########
            num_classes = []
            for each_target in target:
                if train_small[each_target].dtype == 'object' or str(train_small[target[0]].dtype).lower() == 'category':
                    label_encode_flag = True
                elif 0 not in np.unique(train_small[each_target]):
                    label_encode_flag = True
                target_vocab = train_small[each_target].unique().tolist()
                num_classes.append(len(target_vocab))
    else:
        num_classes = 1
        target_vocab = []
    #### This is where we set the model_options for num_classes and num_labels #########
    model_options['num_classes'] = num_classes

    #############   Sample Data classifying features into variaous types ##################  
    print('Loaded a small data sample of size = %s into pandas dataframe to analyze...' %(train_small.shape,))
    ### classify variables using the small dataframe ##
    print('    Classifying variables using data sample in pandas...')
    var_df1, cat_vocab_dict = classify_features_using_pandas(train_small, target, model_options, verbose=verbose)
    
    model_options['modeltype'] = modeltype
    model_options['model_label'] = model_label
    cat_vocab_dict['modeltype'] = modeltype
    cat_vocab_dict['target_variables'] = usecols
    cat_vocab_dict['num_classes'] = num_classes
    cat_vocab_dict["target_transformed"] = label_encode_flag

    # Construct a lookup table to map string chars to indexes,
    
    # using the vocab loaded above:
    if label_encode_flag:
        #### Sometimes, using tf.int64 works. Hence this is needed.
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=target_vocab, values=tf.constant(list(range(len(target_vocab))),
                                               dtype=tf.int64)),
            default_value=int(len(target_vocab)+1))

        trans_output, cat_vocab_dict = transform_train_target(train_small, target, modeltype, 
                                    model_label, cat_vocab_dict)
        train_small[target] = trans_output.values
    
    #### Set column defaults while reading dataset from CSV files - that way, missing values avoided!
    ### The following are valid CSV dtypes for missing values: float32, float64, int32, int64, or string
    ### fill all missing values in categorical variables with "None"
    ### Similarly. fill all missing values in float variables with -99
    if train_small.isnull().sum().sum() > 0:
        print('There are %d missing values in dataset: filling them with default values...' %(
                                train_small.isnull().sum().sum()))
    string_cols = train_small.select_dtypes(include='object').columns.tolist() + train_small.select_dtypes(
                                        include='category').columns.tolist()
    integer_cols =  train_small.select_dtypes(include='integer').columns.tolist()
    float_cols = train_small.select_dtypes(include='float').columns.tolist()
    column_defaults = [-99.0 if x in float_cols else -99 if x in integer_cols else "missing" 
                                for x in list(train_small)]

    #### CAUTION: (num_epochs=None) will automatically repeat the data forever! Be Careful with it!
    ### setting num_epochs to 1 is always good practice since it ensures that your dataset is readable later
    ###  If you set num_epochs to None it will throw your dataset batches into infinite loop. Be careful!
    ####  Also the dataset will display the batch size as 4 (or whatever) if you set num_epochs as None.
    ####  However, if you set num_epochs=1, then you will see dataset shape as None!
    ####  Also num_epochs=1 need to do repeat() on the dataset to loop it forever.
    num_epochs = 1

    ########### find the number of labels in data ####
    if isinstance(target, str):
        num_labels = 1
    elif isinstance(target, list):
        if len(target) == 1:
            num_labels = 1
        else:
            num_labels = len(target)
    cat_vocab_dict['num_labels'] = num_labels
    model_options['num_labels'] = num_labels

    ###  Initially set this batch_size low so that you can do better model training with small batches ###
    #### It is a good idea to shuffle and batch it with small batch size like 4 immediately ###
    if http_url:
        ### Once a file is in gzip format, you have to load it into pandas and then find file size and batch
        cat_vocab_dict["DS_LEN"] = train_small.shape[0]
        model_options['DS_LEN'] = train_small.shape[0]
        DS_LEN = train_small.shape[0]
    try:
        keras_options["batchsize"] = batch_size
        if isinstance(keras_options["batchsize"], str):
            batch_size = find_batch_size(DS_LEN)
        cat_vocab_dict['batch_size'] = batch_size
    except:
        batch_size = find_batch_size(DS_LEN)
        keras_options["batchsize"] = batch_size
        cat_vocab_dict['batch_size'] = batch_size        
    ######  Do this for selecting what columns to load into TF.Data  #######
    #### This means it is not a test dataset - hence it has target columns - load it too!
    if isinstance(target, str):
        if target == '':
            target_name = None
        else:
            target_name = copy.deepcopy(target)
    elif isinstance(target, list):
        #### then it is a multi-label problem
        target_name = None
    else:
        print('Error: Target %s type not understood' %type(target))
        return

    ################    T F  D A T A   D A T A S E T   L O A D I N G     H E R E ################
    ############    Create a Tensorflow Dataset using the make_csv function #####################
    if http_url:
        print('Since input is http URL file we load it into pandas and then tf.data.Dataset...')
        ### Now load the URL file loaded into pandas into a tf.data.dataset  #############
        if isinstance(target, str):
            if target != '':
                labels = train_small.pop(target)
                data_batches = tf.data.Dataset.from_tensor_slices((dict(train_small), labels))
            else:
                print('target variable is blank - please fix input and try again')
                return
        elif isinstance(target, list):
            ##### For multi-label problems, you need to use dict of labels as well ###
            labels = train_small.pop(target)
            data_batches = tf.data.Dataset.from_tensor_slices((dict(train_small), dict(labels)))
        else:
            data_batches = tf.data.Dataset.from_tensor_slices(dict(train_small))
        ### batch it if you are creating it from a dataframe
        data_batches = data_batches.batch(batch_size, drop_remainder=True)
    else:
        print('Loading your input file(s) data directly into tf.data.Dataset...')
        data_batches = tf.data.experimental.make_csv_dataset(train_datafile,
                                       batch_size=batch_size,
                                       column_names=sel_preds,
                                       label_name=target_name,
                                       num_epochs = num_epochs,
                                       column_defaults=column_defaults,
                                       compression_type=compression_type,
                                       shuffle=shuffle_flag,
                                       num_parallel_reads=tf.data.experimental.AUTOTUNE)
        ############### Additional Checkes needed - do it here  ######
        if num_labels > 1:
            ############### Do this step only for Multi_Label problems ######        
            data_batches = data_batches.map(lambda x: split_combined_ds_into_two(x, usecols, preds))
        if label_encode_flag:
            print('    target label encoding now...')
            data_batches = data_batches.map(lambda x, y: to_ids(x, y, table))
            print('    target label encoding completed.')
    print('    train data loaded successfully.')
    
    drop_cols = var_df1['cols_delete']
    preds = [x for x in list(train_small) if x not in usecols+drop_cols]
    print('\nNumber of predictors to be used = %s in next step: keras preprocessing...' %len(preds))
    cat_vocab_dict['columns_deleted'] = drop_cols
    if len(drop_cols) > 0: ### drop cols that have been identified for deletion ###
        print('Dropping %s columns marked for deletion...' %drop_cols)
        train_small.drop(drop_cols,axis=1,inplace=True)

    return train_small, data_batches, var_df1, cat_vocab_dict, keras_options, model_options
############################################################################################
def to_ids(features, labels, table):
    labels = tf.cast(labels, tf.int64)
    labels = table.lookup(labels)
    return (features, labels)
#############################################################################################
def lenopenreadlines(filename):
    with open(filename) as f:
        return len(f.readlines())
#########################################################################################
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
        batch_ratio = 0.0001
    batch_len = int(batch_ratio*DS_LEN)
    #print('    Batch size selected as %d' %batch_len)
    return batch_len
#########################################################################################
def fill_missing_values_for_TF2(train_small, var_df):
    """
    ########################################################################################
    ### As of now (TF 2.4.1) we still cannot load pd.dataframe with nulls in string columns!
    ### You must first remove nulls from the objects in dataframe and use that sample
    ### to build a normalizer layer. You can use Mean and SD from that sample.
    ### Using that sample, you can build the layer for complete dataset
    #### in that case the input is a dataframe, you must first remove nulls from it
    ########################################################################################
    ### Filling Strategy (this is not Imputation - mind you)
    ###    1. Fill all missing values in categorical variables with "None"
    ###    2. Similarly, fill all missing values in float variables with -99
    ########################################################################################
    """
    train_small = copy.deepcopy(train_small)
    cols_delete = var_df['cols_delete']
    cat_cols = var_df['categorical_vars'] + var_df['discrete_string_vars']
    int_cats = var_df['int_cats']
    ints = var_df['int_vars']
    int_cols = int_cats + ints
    float_cols = var_df['continuous_vars']
    nlp_cols = var_df['nlp_vars']
    date_vars = var_df['date_vars']
    lats = var_df['lats']
    lons = var_df['lons']
    ffill_cols = lats + lons + date_vars + cols_delete

    if len(cat_cols) > 0:
        if train_small[cat_cols].isnull().sum().sum() > 0:
            for col in cat_cols:
                colcount = "Missing"
                train_small[col].fillna(colcount, inplace=True)

    if len(nlp_cols) > 0:
        if train_small[nlp_cols].isnull().sum().sum() > 0:
            for col in nlp_cols:
                colcount = "Missing"
                train_small[col].fillna(colcount, inplace=True)

    ints_copy = int_cols + int_cats
    if len(ints_copy) > 0:
        if train_small[ints_copy].isnull().sum().sum() > 0:
            for col in ints_copy:
                colcount = 0
                train_small[col].fillna(colcount,inplace=True)

    if len(float_cols) > 0:
        if train_small[float_cols].isnull().sum().sum() > 0:
            for col in float_cols:
                colcount = 0.0
                train_small[col].fillna(colcount,inplace=True)

    if len(ffill_cols) > 0:
        ffill_cols_copy = copy.deepcopy(ffill_cols)
        if train_small[ffill_cols].isnull().sum().sum() > 0:
            for col in ffill_cols:
                train_small[col].fillna(method='ffill', inplace=True)
        #### Sometimes forward-fill doesn't do it. You need to try back-fill too!
        if train_small[ffill_cols].isnull().sum().sum() > 0:
            for col in ffill_cols_copy:
                train_small[col].fillna(method='bfill', inplace=True)
    return train_small
########################################################################################
def load_train_data_frame(train_small, target, keras_options, model_options, verbose=0):
    """
    ### CAUTION: TF2.4 Still cannot load a DataFrame with Nulls in string or categoricals!
    ############################################################################
    #### TF 2.4 still cannot load tensor_slices into ds if an object or string column
    ####   that has nulls in it! So we need to find other ways to load tensor_slices by
    ####   first filling dataframe with pandas fillna() function!
    #############################################################################
    """
    DS_LEN = model_options['DS_LEN']
    #### do this for dataframes ##################
    try:
        batch_size = keras_options["batchsize"]
        if isinstance(keras_options["batchsize"], str):
            batch_size = find_batch_size(DS_LEN)
    except:
        #### If it is not given find it here ####
        batch_size = find_batch_size(DS_LEN)
    #########  Modify or Convert column names to fit tensorflow rules of no space in names!
    sel_preds = ["_".join(x.split(" ")) for x in list(train_small) ]
    train_small.columns = sel_preds
    if isinstance(target, str):
        target = "_".join(target.split(" "))
        model_label = 'Single_Label'
    else:
        target = ["_".join(x.split(" ")) for x in target ]
        model_label = 'Multi_Label'
    print('    Modified file names to fit no-space in column names rule in Tensorflow!')

    ### usecols is basically target in a list format. Very handy to know when target is a list.
        
    try:
        modeltype = model_options["modeltype"]
        if model_options["modeltype"] == '':
            ### usecols is basically target in a list format. Very handy to know when target is a list.
            modeltype, model_label, usecols = find_problem_type(train_small, target, model_options, verbose)
        else:
            if isinstance(target, str):
                usecols = [target]
            else:
                usecols = copy.deepcopy(target)
    except:
        ### if modeltype is given, then do not find the model type using this function
        modeltype,  model_label, usecols = find_problem_type(train_small, target, model_options, verbose)
    
    ###   Cat_Vocab_Dict contains all info about vocabulary in each variable and their size
    print('    Classifying variables using data sample in pandas...')
    var_df, cat_vocab_dict = classify_features_using_pandas(train_small, target, model_options, verbose=verbose)

    model_options['modeltype'] = modeltype
    model_options['model_label'] = model_label
    cat_vocab_dict['target_variables'] = usecols
    cat_vocab_dict['modeltype'] = modeltype
    model_options['batch_size'] = batch_size

    ##########  Find small details about the data to help create the right model ###
    target_transformed = False
    if modeltype != 'Regression':
        if isinstance(target, str):
            #### This is for Single Label Problems ######
            if train_small[target].dtype == 'object' or str(train_small[target].dtype).lower() == 'category':
                target_transformed = True
                target_vocab = train_small[target].unique()
                num_classes = len(target_vocab)
            else:
                if 0 not in np.unique(train_small[target]):
                    target_transformed = True ### label encoding must be done since no zero class!
                    target_vocab = train_small[target].unique()
                num_classes = len(train_small[target].value_counts())
        elif isinstance(target, list):
            #### This is for Multi-Label Problems #######
            copy_target = copy.deepcopy(target)
            num_classes = []
            for each_target in copy_target:
                if train_small[target[0]].dtype == 'object' or str(train_small[target[0]].dtype).lower() == 'category':
                    target_transformed = True
                    target_vocab = train_small[target].unique().tolist()
                    num_classes_each = len(target_vocab)
                else:
                    if 0 not in np.unique(train_small[target[0]]):
                        target_transformed = True ### label encoding must be done since no zero class!
                        target_vocab = train_small[target[0]].unique()
                    num_classes_each = train_small[target].apply(np.unique).apply(len).max()
                num_classes.append(int(num_classes_each))
    else:
        num_classes = 1
        target_vocab = []
    ########### find the number of labels in data ####
    if isinstance(target, str):
        num_labels = 1
    elif isinstance(target, list):
        if len(target) == 1:
            num_labels = 1
        else:
            num_labels = len(target)
    #### This is where we set the model_options for num_classes and num_labels #########
    model_options['num_labels'] = num_labels
    model_options['num_classes'] = num_classes
    cat_vocab_dict['num_labels'] = num_labels
    cat_vocab_dict['num_classes'] = num_classes
    cat_vocab_dict["target_transformed"] = target_transformed

    ####   fill missing values using this function ##############
    train_small = fill_missing_values_for_TF2(train_small, var_df)
    
    ##### Do the deletion of cols after filling with missing values since otherwise fill errors!
    drop_cols = var_df['cols_delete']
    cat_vocab_dict['columns_deleted'] = drop_cols
    if len(drop_cols) > 0: ### drop cols that have been identified for deletion ###
        print('    Dropping %s columns marked for deletion...' %drop_cols)
        train_small.drop(drop_cols,axis=1,inplace=True)

    ######### Now load the train Dataframe into a tf.data.dataset  #############
    if target_transformed:
        ####################### T R A N S F O R M I N G   T A R G E T ########################
        train_small[target], cat_vocab_dict = transform_train_target(train_small, target, modeltype, 
                                                model_label, cat_vocab_dict)

    if isinstance(target, str):
        #### For single label do this: labels can be without names since there is only one label
        if target != '':
            labels = train_small[target]
            features = train_small.drop(target, axis=1)
            ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        else:
            print('target variable is blank - please fix input and try again')
            return
    elif isinstance(target, list):
        #### For multi label do this: labels must be dict and hence with names since there are many targets
            labels = train_small[target]
            features = train_small.drop(target, axis=1)
            ds = tf.data.Dataset.from_tensor_slices((dict(features), dict(labels)))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(train_small))
    ### batch it if you are creating it from a dataframe
    ds = ds.batch(batch_size, drop_remainder=True)

    try:
        keras_options["batchsize"] = batch_size
        cat_vocab_dict['batch_size'] = batch_size
    except:
        batch_size = find_batch_size(DS_LEN)
        keras_options["batchsize"] = batch_size
        cat_vocab_dict['batch_size'] = batch_size        

    return train_small, ds, var_df, cat_vocab_dict, keras_options, model_options
###############################################################################################
def load_image_data(train_data_or_file, target, project_name, keras_options, model_options,
                        verbose=0):
    """
    Handy function that collects a sequence of image files  into a tf.data generator.

    Inputs:
    -----------
    train_data_or_file: this can be a name of file to load or can be a pandas dataframe to load into tf.data
                  either option will work. This function will detect that automatically and load them.
    target: target name as a string or a 
    """
    cat_vocab_dict = dict()
    cat_vocab_dict['target_variables'] =  target
    cat_vocab_dict['project_name'] = project_name
    if 'image_directory' in model_options.keys():
        print('    Image directory given as %s' %model_options['image_directory'])
        image_directory = model_options["image_directory"]
    else:
        print("    No image directory given. Provide image directory in model_options...")
        return
    if 'image_height' in model_options.keys():
        print('    Image height given as %d' %model_options['image_height'])
    else:
        print("    No image height given. Returning. Provide image height and width...")
        return
    if 'image_width' in model_options.keys():
        print('    Image width given as %d' %model_options['image_width'])
    else:
        print("    No image width given. Returning. Provide image height and width...")
        return
    if 'image_channels' in model_options.keys():
        print('    Image channels given as %d' %model_options['image_channels'])
    else:
        print("    No image_channels given. Returning. Provide image height and width...")
        return
    try:
        image_train_folder = os.path.join(image_directory,"train")
        if not os.path.exists(image_train_folder):
            print("Image use case. No train folder exists under given directory. Returning...")
            return
    except:
        print('Error: Not able to find any train or test image folder in the given folder %s' %image_train_folder)
        print("""   You must put images under folders named train, 
                    validation (optional) and test folders under given %s folder.
                    Otherwise deep_autoviml won't work. """ %image_directory)
    image_train_split = False
    image_train_folder = os.path.join(image_directory,"train")
    if not os.path.exists(image_train_folder):
        print("No train folder found under given image directory %s. Returning..." %image_directory)
        image_train_folder = os.path.join(image_directory,"validation")
    image_valid_folder = os.path.join(image_directory,"validation")
    if not os.path.exists(image_valid_folder):
        print("No validation folder found under given image directory %s. Returning..." %image_directory)
        image_train_split = True
    img_height = model_options['image_height']
    img_width = model_options['image_width']
    img_channels = model_options['image_channels']
    #### make this a small number - default batch_size ###
    batch_size = check_model_options(model_options, "batch_size", 64)
    model_options["batch_size"] = batch_size
    full_ds = tf.keras.preprocessing.image_dataset_from_directory(image_train_folder,
                                  seed=111,
                                  image_size=(img_height, img_width),
                                  batch_size=batch_size)
    if image_train_split:
        ############## Split train into train and validation datasets here ###############
        classes = full_ds.class_names
        recover = lambda x,y: y
        print('\nSplitting train into two: train and validation data')
        valid_ds = full_ds.enumerate().filter(is_valid).map(recover)
        train_ds = full_ds.enumerate().filter(is_train).map(recover)
    else:
        train_ds = full_ds
        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(image_valid_folder,
          seed=111,
          image_size=(img_height, img_width),
          batch_size=batch_size)
        classes = train_ds.class_names
    ####  Successfully loaded train and validation data sets ################
    cat_vocab_dict["image_classes"] = classes
    cat_vocab_dict["target_transformed"] = True
    cat_vocab_dict['modeltype'] =  'Classification'
    MLB = My_LabelEncoder()
    ins = copy.deepcopy(classes)
    outs = np.arange(len(classes))
    MLB.transformer = dict(zip(ins,outs))
    MLB.inverse_transformer = dict(zip(outs,ins))
    cat_vocab_dict['target_le'] = MLB
    print('Number of image classes = %d and they are: %s' %(len(classes), classes))
    print_one_image_from_dataset(train_ds, classes)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    cat_vocab_dict["image_height"] = img_height
    cat_vocab_dict["image_width"] = img_width
    cat_vocab_dict["batch_size"] = batch_size
    cat_vocab_dict["image_channels"] = img_channels
    return train_ds, valid_ds, cat_vocab_dict, model_options
########################################################################################
from collections import defaultdict
from collections import Counter

def load_train_data(train_data_or_file, target, project_name, keras_options, model_options,
                  verbose=0):
    """
    Handy function that loads a file or a sequence of files (*.csv) into a tf.data.Dataset 
    You can also load a pandas dataframe instead of a file if you wanted to. It accepts both!
    It will automatically figure out whether input is a file or file(s) or a pandas dataframe.

    Inputs:
    -----------
    train_data_or_file: this can be a name of file to load or can be a pandas dataframe to load into tf.data
                  either option will work. This function will detect that automatically and load them.
    target: target name as a string or a 
    """
    shuffle_flag = False
    cat_vocab_dict = defaultdict(list)
    maxrows = 10000 ### the number of maximum rows read by pandas to sample data ##
    ### Since you cannot deal with a very large dataset in pandas, let's look into how big the file is
    try:
        if isinstance(train_data_or_file, str):
            DS_LEN = lenopenreadlines(train_data_or_file)
        else:
            DS_LEN = train_data_or_file.shape[0]
    except:
        if find_words_in_list(['http'], [train_data_or_file.lower()]):
            print('http url file: cannot find size of dataset. Setting default...')
        DS_LEN = maxrows #### set to an artificial low number ###
    keras_options["data_size"] = DS_LEN
    model_options["DS_LEN"] = DS_LEN
    ##########    LOADING EITHER FILE OR DATAFRAME INTO TF DATASET HERE  ##################
    if isinstance(train_data_or_file, str):
        #### do this for files only ##################
        train_small, ds, var_df, cat_vocab_dict, keras_options, model_options = load_train_data_file(train_data_or_file, target, 
                                                                keras_options, model_options, verbose)
    else:
        train_small, ds, var_df, cat_vocab_dict, keras_options, model_options = load_train_data_frame(train_data_or_file, target,
                                                                keras_options, model_options, verbose)
    cat_vocab_dict['DS_LEN'] = DS_LEN
    if verbose >= 1 and train_small.shape[1] <= 30:
        print_one_row_from_tf_dataset(ds)
    ####  Set Class Weights for Imbalanced Data Sets here ##########
    modeltype = model_options["modeltype"]
    #### You need to do this transform only for files. Otherwise, it is done already for dataframes.
    if isinstance(target, str):
        y_train = train_small[target]
        if modeltype != 'Regression' and not cat_vocab_dict['target_transformed']:
            cat_vocab_dict["original_classes"] = np.unique(train_small[target])
    else:
        y_train = train_small[target[0]]
        target_copy = copy.deepcopy(target)
        if modeltype != 'Regression' and not cat_vocab_dict['target_transformed']:
            for each_t in target_copy:
                cat_vocab_dict[each_t+"_original_classes"] = np.unique(train_small[each_t])
    ####  CREATE  CLASS_WEIGHTS HERE #################
    if modeltype != 'Regression':
        find_rare_class(y_train, verbose=1)
        if keras_options['class_weight'] and not model_options['model_label']=='Multi_Label':
            # Class weights are only applicable to single labels in Keras right now
            class_weights = get_class_distribution(y_train)
            keras_options['class_weight'] = class_weights
            print('    Class weights calculated: %s' %class_weights)
        else:
            keras_options['class_weight'] = {}
    else:
        keras_options['class_weight'] = {}
        print('    No class weights specified. Continuing...')
    return train_small, model_options, ds, var_df, cat_vocab_dict, keras_options
##########################################################################################################
from collections import OrderedDict
def find_rare_class(classes, verbose=0):
    ######### Print the % count of each class in a Target variable  #####
    """
    Works on Multi Class too. Prints class percentages count of target variable.
    It returns the name of the Rare class (the one with the minimum class member count).
    This can also be helpful in using it as pos_label in Binary and Multi Class problems.
    """
    counts = OrderedDict(Counter(classes))
    total = sum(counts.values())
    if verbose >= 1:
        print('       Class  -> Counts -> Percent')
        sorted_keys = sorted(counts.keys())
        for cls in sorted_keys:
            print("%12s: % 7d  ->  % 5.1f%%" % (cls, counts[cls], counts[cls]/total*100))
    if type(pd.Series(counts).idxmin())==str:
        return pd.Series(counts).idxmin()
    else:
        return int(pd.Series(counts).idxmin())
###############################################################################
from sklearn.utils.class_weight import compute_class_weight
import copy
from collections import Counter
def get_class_distribution(y_input):
    y_input = copy.deepcopy(y_input)
    classes = np.unique(y_input)
    xp = Counter(y_input)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_input), y=y_input)
    if len(class_weights[(class_weights> 10)]) > 0:
        class_weights = (class_weights/10)
    else:
        class_weights = (class_weights)
    #print('    class_weights = %s' %class_weights)
    class_weights[(class_weights<1)]=1
    class_rows = class_weights*[xp[x] for x in classes]
    class_rows = class_rows.astype(int)
    class_weighted_rows = dict(zip(classes,class_weights))
    #print('    class_weighted_rows = %s' %class_weighted_rows)
    return class_weighted_rows
########################################################################
### Split raw_train_set into train and valid data sets first
### This is a better way to split a dataset into train and test ####
### It does not assume a pre-defined size for the data set.
def is_valid(x, y):
    return x % 5 == 0
def is_test(x, y):
    return x % 2 == 0
def is_train(x, y):
    return not is_test(x, y)
##################################################################################
