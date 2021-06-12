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
#####################################################################################
# Utils
from deep_autoviml.utilities.utilities import print_one_row_from_tf_dataset, print_one_row_from_tf_label
from deep_autoviml.utilities.utilities import print_classification_metrics, print_regression_model_stats
from deep_autoviml.utilities.utilities import print_classification_model_stats, plot_history, plot_classification_results
from deep_autoviml.utilities.utilities import save_valid_predictions, print_classification_header
from deep_autoviml.utilities.utilities import get_callbacks, get_chosen_callback
from deep_autoviml.modeling.create_model import check_keras_options

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
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
import math
#########################################################################################
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
def train_model(deep_model, full_ds, target, keras_model_type, keras_options,
                 model_options, var_df, cat_vocab_dict, project_name="", save_model_flag=True, 
                 verbose=0 ):
    """
    Given a keras model and a tf.data.dataset that is batched, this function will 
    train a keras model. It will first split the batched_data into train_ds and  
    valid_ds (80/20). Then it will select the right parameters based on model type and 
    train the model and evaluate it on valid_ds. It will return a keras model fully 
    trained on the full batched_data finally and train history.
    """
    ####  just use modeltype for printing that's all ###
    start_time = time.time()
    ### check the defaults for the following!
    save_weights_only = check_keras_options(keras_options, "save_weights_only", False)
    data_size = check_keras_options(keras_options, 'data_size', 10000)
    batch_size = check_keras_options(keras_options, 'batchsize', 64)
    num_classes = model_options["num_classes"]
    num_labels = model_options["num_labels"]
    modeltype = model_options["modeltype"]
    patience = check_keras_options(keras_options, "patience", 10)
    class_weights = check_keras_options(keras_options, "class_weight", {})
    print('    class_weights: %s' %class_weights)
    cols_len = len([item for sublist in list(var_df.values()) for item in sublist])
    print('    original datasize = %s, initial batchsize = %s' %(data_size, batch_size))
    NUMBER_OF_EPOCHS = check_keras_options(keras_options, "epochs", 100)
    learning_rate = 5e-1
    steps = max(10, 1*(data_size//batch_size))
    print('    recommended steps per epoch = %d' %steps)
    onecycle_steps = math.ceil(data_size / batch_size) * NUMBER_OF_EPOCHS
    print('    recommended OneCycle steps = %d' %onecycle_steps)
    STEPS_PER_EPOCH = check_keras_options(keras_options, "steps_per_epoch", 
                        steps)
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    #keras.optimizers.schedules.ExponentialDecay(0.01,STEPS_PER_EPOCH, 0.95)
    #### These can be standard for every keras option that you use layers ######
    kernel_initializer = check_keras_options(keras_options, 'kernel_initializer', 'lecun_normal')
    activation='selu'
    print('    default initializer = %s, default activation = %s' %(kernel_initializer, activation))
    #####   set some defaults for model parameters here ##
    optimizer = check_keras_options(keras_options,'optimizer', Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999))
    #optimizer = SGD(lr=learning_rate, momentum = 0.9)
    print('    Using optimizer = %s' %str(optimizer).split(".")[-1][:8])
    use_bias = check_keras_options(keras_options, 'use_bias', True)
    val_monitor = keras_options['monitor']
    val_mode = keras_options['mode']
    patience = keras_options["patience"]

    if keras_options['lr_scheduler'] in ['',"onecycle", "onecycle2"]:
        #### you need to double the amount of patience for onecycle scheduler ##
        print('    Increasing the amount of patience for onecycle scheduler')
        patience = patience * 1.2
    callbacks_dict, tb_logpath = get_callbacks(val_mode, val_monitor, patience, learning_rate, 
                            save_weights_only, onecycle_steps)
    chosen_callback = get_chosen_callback(callbacks_dict, keras_options)

    if keras_options['early_stopping']:
        callbacks_list = [chosen_callback,  callbacks_dict['tb'],  callbacks_dict['pr'], callbacks_dict['es']]
    else:
        callbacks_list = [chosen_callback, callbacks_dict['tb'],  callbacks_dict['pr']]

    print('    val mode = %s, val monitor = %s, patience = %s' %(val_mode, val_monitor, patience))
    print('    number of epochs = %d, steps per epoch = %d' %(NUMBER_OF_EPOCHS, STEPS_PER_EPOCH))
    ############## Split train into train and validation datasets here ###############
    ##################################################################################
    recover = lambda x,y: y
    print('    Splitting train into two: train and validation data')
    valid_ds1 = full_ds.enumerate().filter(is_valid).map(recover)
    train_ds = full_ds.enumerate().filter(is_train).map(recover)
    heldout_ds1 = valid_ds1
    ##################################################################################
    valid_ds = heldout_ds1.enumerate().filter(is_test).map(recover)
    heldout_ds = heldout_ds1.enumerate().filter(is_test).map(recover)
    print('    Splitting validation into two: valid and heldout data')
    ##################################################################################
    ###   V E R Y    I M P O R T A N T  S T E P   B E F O R E   M O D E L   F I T  ###    
    ##################################################################################
    shuffle_size = int(data_size)
    #shuffle_size = 100000
    print(' Shuffle size = %d' %shuffle_size)
    train_ds = train_ds.prefetch(batch_size).shuffle(shuffle_size, 
                            reshuffle_each_iteration=False, seed=42)#.repeat()
    valid_ds = valid_ds.prefetch(batch_size)#.repeat()
    print('Training %s model now. This will take time...' %keras_model_type)
    
    np.random.seed(42)
    tf.random.set_seed(42)
    history = deep_model.fit(train_ds, validation_data=valid_ds, class_weight=class_weights,
                    epochs=NUMBER_OF_EPOCHS, #steps_per_epoch=STEPS_PER_EPOCH, 
                    callbacks=callbacks_list, #validation_steps=STEPS_PER_EPOCH,
                   shuffle=False)

    print('    Model training metrics available: %s' %history.history.keys())
    try:
        ##### this is where it stopped - you have toi subtract patience from it
        stopped_epoch = int(pd.DataFrame(history.history).shape[0] - patience )
    except:
        stopped_epoch = 100

    print('Time taken to train model (in mins) = %0.0f' %((time.time()-start_time)/60))

    #### train the model on full train data set now ###
    start_time = time.time()
    print('    Stopped epoch = %s' %stopped_epoch)

    ##################################################################################
    #######        S A V E the model here using save_model_name      #################
    ##################################################################################
    
    if isinstance(project_name,str):
        if project_name == '':
            project_name = "deep_autoviml"
    else:
        print('Project name must be a string and helps create a folder to store model.')
        project_name = "deep_autoviml"
    save_model_path = os.path.join(project_name,keras_model_type)
    cat_vocab_dict['project_name'] = project_name

    if save_model_flag:
        print('\nSaving model in %s now...this will take time...' %save_model_path)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        deep_model.save(save_model_path)
        cat_vocab_dict['saved_model_path'] = save_model_path
        print('     deep model saved in %s directory' %save_model_path)
    else:
        print('\nModel not being saved since save_model_flag set to False...')

    #### make sure you save the cat_vocab_dict to use later during predictions
    try:
        pickle_path = os.path.join(project_name,"cat_vocab_dict")+".pickle"
        if not os.path.exists(project_name):
            os.makedirs(project_name)
        print('\nSaving vocab dictionary using pickle in %s...will take time...' %pickle_path)
        with open(pickle_path, "wb") as fileopen:
            fileopen.write(pickle.dumps(cat_vocab_dict))
        print('    Saved pickle file in %s' %pickle_path)
    except:
        print('Unable to save cat_vocab_dict - please pickle it yourself.')
    ####### make sure you save the variable definitions file ###########
    try:
        pickle_path = os.path.join(project_name,"var_df")+".pickle"
        if not os.path.exists(project_name):
            os.makedirs(project_name)
        print('\nSaving variable definitions file using pickle in %s...will take time...' %pickle_path)
        with open(pickle_path, "wb") as fileopen:
            fileopen.write(pickle.dumps(var_df))
        print('    Saved pickle file in %s' %pickle_path)
    except:
        print('Unable to save cat_vocab_dict - please pickle it yourself.')
    
    #################################################################################
    ########     P R E D I C T   O N   H E L D   O U T  D A T A   H E R E      ######
    #################################################################################
    
    if num_labels <= 1:
        y_test = np.concatenate(list(heldout_ds.map(lambda x,y: y).as_numpy_iterator()))
        print('Single-Label: Heldout data shape: %s' %(y_test.shape,))
    else:
        iters = int(data_size/batch_size) + 1
        for inum, each_target in enumerate(target):
            add_ls = []
            for feats, labs in heldout_ds.take(iters): 
                add_ls.append(list(labs[each_target].numpy()))
            flat_list = [item for sublist in add_ls for item in sublist]
            if inum == 0:
                each_array = np.array(flat_list)
            else:
                each_array = np.c_[each_array, np.array(flat_list)]
        y_test = copy.deepcopy(each_array)
        print('Multi-Label: Heldout data shape: %s' %(y_test.shape,))
    scores = []
    ls = []
    if verbose >= 1:
        print_one_row_from_tf_label(heldout_ds)
    ###########################################################################
    y_probas = deep_model.predict(heldout_ds)
    
    if isinstance(target, str):
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
            for each_t in range(len(target)):
                if each_t == 0:
                    y_test_preds = y_probas[each_t].argmax(axis=1).astype(int)
                else:
                    y_test_preds = np.c_[y_test_preds, y_probas[each_t].argmax(axis=1).astype(int)]
        else:
            ### This is for Multi-Label Regression ###
            for each_t in range(len(target)):
                if each_t == 0:
                    y_test_preds = y_probas[each_t].mean(axis=1)
                else:
                    y_test_preds = np.c_[y_test_preds, y_probas[each_t].mean(axis=1)]
                if y_test.dtype == 'int':
                    y_test_preds = y_test_preds.round().astype(int)

    print('\nHeld out predictions shape:%s' %(y_test_preds.shape,))
    if verbose >= 1:
        if modeltype != 'Regression':
            print('    Sample predictions: %s' %y_test_preds[:10])
        else:
            if num_labels == 1:
                print('    Sample predictions: %s' %y_test_preds.ravel()[:10])
            else:
                print('    Sample predictions:\n%s' %y_test_preds[:10])

    #################################################################################
    ########     P L O T T I N G   V A L I D A T I O N   R E S U L T S         ######
    #################################################################################
    ###  Plot the epochs and loss metrics here #####################    
    try:
        #print('    Additionally, Tensorboard logs can be found here: %s' %tb_logpath)
        if modeltype == 'Regression':
            plot_history(history, val_monitor[4:], target)
        elif modeltype == 'Classification':
            plot_history(history, val_monitor[4:], target)
        else:
            plot_history(history, val_monitor[4:], target)
    except:
        print('    Plot history is erroring. Tensorboard logs can be found here: %s' %tb_logpath)

    print('\n###########################################################')
    print('         Held-out test data set Results:')
    num_labels = cat_vocab_dict['num_labels']
    num_classes = cat_vocab_dict['num_classes']
    if num_labels <= 1:
        #### This is for Single-Label Problems only ################################
        if modeltype == 'Regression':
            print_regression_model_stats(y_test, y_test_preds,target,plot_name=project_name)
        else:
            print_classification_header(num_classes, num_labels, target)
            labels = cat_vocab_dict['original_classes']
            if cat_vocab_dict['target_transformed']:
                target_names = cat_vocab_dict['transformed_classes']
                target_le = cat_vocab_dict['target_le']
                y_pred = y_probas.argmax(axis=1)
                y_test_trans = target_le.inverse_transform(y_test)
                y_pred_trans = target_le.inverse_transform(y_pred)
                plot_classification_results(y_test_trans, y_pred_trans, labels, labels, target)
            else:
                y_pred = y_probas.argmax(axis=1)
                plot_classification_results(y_test, y_pred, labels, labels, target)
            print_classification_metrics(y_test, y_probas, proba_flag=True)
    else:
        if modeltype == 'Regression':
            #### This is for Multi-Label Regression ################################
            print_regression_model_stats(y_test, y_test_preds,target,plot_name=project_name)
        else:
            #### This is for Multi-Label Classification ################################
            try:
                targets = cat_vocab_dict["target_variables"]
                for i, each_target in enumerate(targets):
                    print_classification_header(num_classes, num_labels, each_target)
                    labels = cat_vocab_dict[each_target+'_original_classes']
                    if cat_vocab_dict['target_transformed']:
                        ###### Use a nice classification matrix printing module here #########
                        target_names = cat_vocab_dict[each_target+'_transformed_classes']
                        target_le = cat_vocab_dict['target_le'][i]
                        y_pred = y_probas[i].argmax(axis=1)
                        y_test_trans = target_le.inverse_transform(y_test[:,i])
                        y_pred_trans = target_le.inverse_transform(y_pred)
                        labels = np.unique(y_test_trans) ### sometimes there is less classes
                        plot_classification_results(y_test_trans, y_pred_trans, labels, labels, each_target)
                    else:
                        y_pred = y_probas[i].argmax(axis=1)
                        labels = np.unique(y_test[:,i]) ### sometimes there are fewer classes ##
                        plot_classification_results(y_test[:,i], y_pred, labels, labels, each_target)
                    print_classification_metrics(y_test[:,i], y_probas[i], proba_flag=True)
                    #### This prints additional metrics #############
                    print(classification_report(y_test[:,i],y_test_preds[:,i]))
                    print(confusion_matrix(y_test[:,i], y_test_preds[:,i]))
            except:
                print_classification_metrics(y_test, y_test_preds, False)
                print(classification_report(y_test, y_test_preds ))
    ### plot the regression results here #########
    if modeltype == 'Regression':
        if isinstance(target, str):
            plt.figure(figsize=(15,6))
            ax1 = plt.subplot(1, 2, 1)
            ax1.scatter(x=y_test, y=y_test_preds,)
            ax1.set_title('Actuals (x-axis) vs. Predictions (y-axis)')
            pdf = save_valid_predictions(y_test, y_test_preds.ravel(), project_name, num_labels)
            ax2 = plt.subplot(1, 2, 2)
            pdf.plot(ax=ax2)
        else:
            pdf = save_valid_predictions(y_test, y_test_preds, project_name, num_labels)
            plt.figure(figsize=(15,6))
            for i in range(num_labels):
                ax1 = plt.subplot(1, num_labels, i+1)
                ax1.scatter(x=y_test[:,i], y=y_test_preds[:,i])
                ax1.set_title(f"Actuals_{i} (x-axis) vs. Predictions_{i} (y-axis)")
            plt.figure(figsize=(15, 6)) 
            for j in range(num_labels):
                pair_cols = ['actuals_'+str(j), 'predictions_'+str(j)]
                ax2 = plt.subplot(1, num_labels, j+1)
                pdf[pair_cols].plot(ax=ax2)

    ##################################################################################
    ###   V E R Y    I M P O R T A N T  S T E P   B E F O R E   M O D E L   F I T  ###    
    ##################################################################################
    print('\nTraining full train dataset for %d epochs. This will take time...' %stopped_epoch)
    full_ds = full_ds.shuffle(shuffle_size).prefetch(batch_size) #.repeat()
    deep_model.fit(full_ds, epochs=stopped_epoch, #steps_per_epoch=STEPS_PER_EPOCH,
                 class_weight=class_weights, verbose=0)

    print('    completed. Time taken (in mins) = %0.0f' %((time.time()-start_time)/100))

    return deep_model, cat_vocab_dict
######################################################################################
