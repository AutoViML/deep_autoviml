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
import pdb
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings(action='ignore')
import functools
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
################################################################################
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
from tensorflow.keras.models import Model, load_model
################################################################################
from deep_autoviml.modeling.one_cycle import OneCycleScheduler

################################################################################
import os
def check_if_GPU_exists(verbose=0):
    GPU_exists = False
    gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    tpus = tf.config.list_logical_devices('TPU')
    #### In some cases like Kaggle kernels, the GPU is not enabled. Hence this check.
    if logical_gpus:
        # Restrict TensorFlow to only use the first GPU        
        if verbose:
            print("Num GPUs Available: ", len(logical_gpus))
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        if len(logical_gpus) > 1:
            device = "gpus"
        else:
            device = "gpu"
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    elif tpus:
        device = "tpu"
        if verbose:
            print("Listing all TPU devices: ")
            for tpu in tpus:
                print(tpu)
    else:
        if verbose:
            print('    Only CPU found on this device')
        device = "cpu"
    #### Set Strategy ##########
    if device == "tpu":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
            tf.config.experimental_connect_to_cluster(resolver)
            # This is the TPU initialization code that has to be at the beginning.
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            if verbose:
                print('Setting TPU strategy using %d devices' %strategy.num_replicas_in_sync)
        except:
            if verbose:
                print('Setting TPU strategy using Colab...')
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            tf.config.experimental_connect_to_cluster(resolver)
            # This is the TPU initialization code that has to be at the beginning.
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
    elif device == "gpu":
        strategy = tf.distribute.MirroredStrategy()
        if verbose:
            print('Setting Mirrored GPU strategy using %d devices' %strategy.num_replicas_in_sync)
    elif device == "gpus":
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        if verbose:
            print('Setting Multiworker GPU strategy using %d devices' %strategy.num_replicas_in_sync)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device='/device:CPU:0')
        if verbose:
            print('Setting CPU strategy using %d devices' %strategy.num_replicas_in_sync)
    return strategy
######################################################################################
def print_one_row_from_tf_dataset(test_ds):
    """
    No matter how big a dataset or batch size, this handy function will print the first row.
    This way you can test what's in each row of a tensorflow dataset that you sent in as input
    You need to provide at least one column in the dataset for it to check if it should print it.
    Inputs:
    -------
    test_ds: tf.data.DataSet - this must be batched and num_epochs must be an integer.
                - otherwise it won't print!
    """
    try:
        if isinstance(test_ds, tuple):
            dict_row = list(test_ds.as_numpy_iterator())[0]
        else:
            dict_row = test_ds
        print("Printing one batch from the dataset:")
        preds = list(dict_row.element_spec[0].keys())
        if dict_row.element_spec[0][preds[0]].shape[0] is None or isinstance(
                dict_row.element_spec[0][preds[0]].shape[0], int):
            for batch, head in dict_row.take(1):
                for labels, value in batch.items():
                    print("{:40s}: {}".format(labels, value.numpy()[:4]))
    except:
        print(list(test_ds.as_numpy_iterator())[0])
#########################################################################################
def print_one_row_from_tf_label(test_label):
    """
    No matter how big a dataset or batch size, this handy function will print the first row.
    This way you can test what's in each row of a tensorflow dataset that you sent in as input
    You need to provide at least one column in the dataset for it to check if it should print it.
    Inputs:
    -------
    test_label: tf.data.DataSet - this must be batched and num_epochs must be an integer.
                - otherwise it won't print!
    """
    if isinstance(test_label, tuple):
        dict_row = list(test_label.as_numpy_iterator())[0]
    else:
        dict_row = test_label
    preds = list(dict_row.element_spec[0].keys())
    try:
        ### This is for multilabel problems only ####
        if len(dict_row.element_spec[1]) >= 1:
            labels = list(dict_row.element_spec[1].keys())
            for feats, labs in dict_row.take(1):
                for each_label in labels:
                    print('    label = %s, samples: %s' %(each_label, labs[each_label]))
    except:
        ### This is for single problems only ####
        if dict_row.element_spec[0][preds[0]].shape[0] is None or isinstance(
                dict_row.element_spec[0][preds[0]].shape[0], int):
            for feats, labs in dict_row.take(1):
                print("    samples from label: %s" %(labs.numpy().tolist()[:10]))
##########################################################################################
from sklearn.base import TransformerMixin
from collections import defaultdict
import pandas as pd
import numpy as np
class My_LabelEncoder(TransformerMixin):
    """
    ################################################################################################
    ######  This Label Encoder class works just like sklearn's Label Encoder!  #####################
    #####  You can label encode any column in a data frame using this new class. But unlike sklearn,
    the beauty of this function is that it can take care of NaN's and unknown (future) values.
    It uses the same fit() and fit_transform() methods of sklearn's LabelEncoder class.
    ################################################################################################
    Usage:
          MLB = My_LabelEncoder()
          train[column] = MLB.fit_transform(train[column])
          test[column] = MLB.transform(test[column])
    """
    def __init__(self):
        self.transformer = defaultdict(str)
        self.inverse_transformer = defaultdict(str)

    def fit(self,testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        outs = np.unique(testx.factorize()[0])
        ins = np.unique(testx.factorize()[1]).tolist()
        if -1 in outs:
            ins.insert(0,np.nan)
        self.transformer = dict(zip(ins,outs.tolist()))
        self.inverse_transformer = dict(zip(outs.tolist(),ins))
        return self

    def transform(self, testx):
        if isinstance(testx, pd.Series):
            pass
        elif isinstance(testx, np.ndarray):
            testx = pd.Series(testx)
        else:
            return testx
        ins = np.unique(testx.factorize()[1]).tolist()
        missing = [x for x in ins if x not in self.transformer.keys()]
        if len(missing) > 0:
            for each_missing in missing:
                max_val = np.max(list(self.transformer.values())) + 1
                self.transformer[each_missing] = max_val
                self.inverse_transformer[max_val] = each_missing
        ### now convert the input to transformer dictionary values
        outs = testx.map(self.transformer).values
        return outs

    def inverse_transform(self, testx):
        ### now convert the input to transformer dictionary values
        if isinstance(testx, pd.Series):
            outs = testx.map(self.inverse_transformer).values
        elif isinstance(testx, np.ndarray):
            outs = pd.Series(testx).map(self.inverse_transformer).values
        else:
            outs = testx[:]
        return outs
#################################################################################
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
#################################################################################
def plot_history(history, metric, targets):
    if isinstance(targets, str):
        #### This is for single label problems
        fig = plt.figure(figsize=(15,6))
        #### first metric is always the loss - just plot it!
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Model Training vs Validation Loss')
        plot_one_history_metric(history, "loss", ax1)
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Model Training vs Validation %s' %metric)
        ##### Now let's plot the second metric ####
        plot_one_history_metric(history, metric, ax2)
    else:
        ### This is for Multi-Label problems
        for each_target in targets:
            fig = plt.figure(figsize=(15,6))
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title('Model Training vs Validation Loss')
            plot_one_history_metric(history, each_target+"_loss", ax1)
            ax2 = plt.subplot(1, 2, 2)
            ### Since we are using total loss, we must find another metric to show. 
            ###  This is how we do it - by collecting all metrics with target name 
            ###  and pick the first one. This may or may not always get the best answer, but we will see.
            metric1 = [x for x in hist.columns.tolist() if (each_target in x) & ("loss" not in x) ]
            metric2 = metric1[0]
            ### the next line is not a typo! it should always be target[0]
            ### since val_monitor is based on the first target's metric only!
            #metric1 = metric.replace(targets[0],'')
            #metric2 = each_target + metric1
            ax2.set_title('Model Training vs Validation %s' %metric2)
            plot_one_history_metric(history, metric2, ax2)
    plt.show();
#######################################################################################
def plot_one_history_metric(history, metric, ax):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    ax.plot(epochs, train_metrics)
    ax.plot(epochs, val_metrics)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)
    ax.legend(["train_"+metric, 'val_'+metric])
####################################################################################
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from collections import OrderedDict
from collections import Counter
def print_classification_model_stats(y_test, y_preds):
    """
    This will print both multi-label and multi-class metrics. 
    You must send in only actual and predicted labels. No probabilities!!
    """
    try:
        assert y_preds.shape[1]
        for i in range(y_preds.shape[1]):
            if y_preds.shape[1] == y_test.shape[1]:
                print('Target label %s results:' %(i+1))
                print_classification_model_metrics(y_test[:,i], y_preds[:,i])
            else:
                print('error printing: number of labels in actuals and predicted are different ')
    except:
        ### This is a binary class only #######
        print_classification_model_metrics(y_test, y_preds)

def print_classification_model_metrics(y_true, predicted):
    """
    This prints classification metrics in a nice format only for binary classes
    """
    #### Use this to Test Classification Problems Only ####
    try:
        y_pred = predicted.argmax(axis=1)
    except:
        y_pred = predicted
    print('Balanced Accuracy = %0.2f%%' %(
        100*balanced_accuracy_score(y_true, y_pred)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print('#####################################################################')
    return balanced_accuracy_score(y_true, y_pred)
###################################################################################
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def plot_classification_results(y_true, y_pred, labels, target_names, title_string=""):
    try:
        fig, axes = plt.subplots(1,2,figsize=(15,6))
        draw_confusion_matrix(y_true, y_pred, labels, target_names, '%s Confusion Matrix' %title_string, ax=axes[0])
        try:
            clf_report = classification_report(y_true,
                                               y_pred,
                                               labels=labels,
                                               target_names=target_names,
                                               output_dict=True)
        except:
            clf_report = classification_report(y_true,y_pred,labels=target_names,
                target_names=labels,output_dict=True)
        sns.heatmap(pd.DataFrame(clf_report).iloc[:, :].T, annot=True,ax=axes[1],fmt='0.2f');
        axes[1].set_title('Classification Report')
    except:
        print('Error: could not plot classification results. Continuing...')
######################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
def draw_confusion_matrix(y_test,y_pred, labels, target_names, model_name='Model',ax=''):
    """
    This plots a beautiful confusion matrix based on input: ground truths and predictions
    """
    #Confusion Matrix
    '''Plotting CONFUSION MATRIX'''
    import seaborn as sns
    sns.set_style('darkgrid')

    '''Display'''
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    pd.options.display.float_format = '{:,.2f}'.format

    #Get the confusion matrix and put it into a df

    cm = confusion_matrix(y_test, y_pred)
    
    cm_df = pd.DataFrame(cm,
                         index = labels,
                         columns = target_names,
                        )

    sns.heatmap(cm_df,
                center=0,
                cmap=sns.diverging_palette(220, 15, as_cmap=True),
                annot=True,
                fmt='g',
               ax=ax)

    ax.set_title(' %s \nF1 Score(avg = micro): %0.2f \nF1 Score(avg = macro): %0.2f' %(
        model_name,f1_score(y_test, y_pred, average='micro'),f1_score(y_test, y_pred, average='macro')),
              fontsize = 13)
    ax.set_ylabel('True label', fontsize = 13)
    ax.set_xlabel('Predicted label', fontsize = 13)
################################################################################
def print_regression_model_stats(actuals, predicted, targets='', plot_name=''):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE to have a title or something, just give that
    in the input as "title" and it will print that title on the MAE and RMSE as a
    chart for that model. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    if isinstance(actuals,pd.Series) or isinstance(actuals,pd.DataFrame):
        actuals = actuals.values
    if isinstance(predicted,pd.Series) or isinstance(predicted,pd.DataFrame):
        predicted = predicted.values
    if len(actuals) != len(predicted):
        print('Error: Number of actuals and predicted dont match. Continuing...')
    if targets == "":
        try:
            ### This is for Multi_Label Problems ###
            assert actuals.shape[1]
            multi_label = True
            if isinstance(actuals,pd.Series):
                cols = [actuals.name]
            elif isinstance(actuals,pd.DataFrame):
                cols = actuals.columns.tolist()
            else:
                cols = ['target_'+str(i) for i in range(actuals.shape[1])]
        except:
            #### THis is for Single Label problems #####
            multi_label = False
            if isinstance(actuals,pd.Series):
                cols = [actuals.name]
            elif isinstance(actuals,pd.DataFrame):
                cols = actuals.columns.tolist()
            else:
                cols = ['target_1']
    else:
        cols = copy.deepcopy(targets)
        if isinstance(targets, str):
            cols = [targets]
        if len(cols) == 1:
            multi_label = False
        else:
            multi_label = True
    try:
        plot_regression_scatters(actuals,predicted,cols,plot_name=plot_name)
    except:
        print('Could not draw regression plot but continuing...')
    if multi_label:
        for i in range(actuals.shape[1]):
            actuals_x = actuals[:,i]
            try:
                predicted_x = predicted[:,i]
            except:
                predicted_x = predicted.ravel()
            print('Regression Metrics for Target=%s' %cols[i])
            mae, mae_asp, rmse_asp = print_regression_metrics(actuals_x, predicted_x)
    else:
        mae, mae_asp, rmse_asp = print_regression_metrics(actuals, predicted)
    return mae, mae_asp, rmse_asp
################################################################################
from sklearn.metrics import r2_score
def print_regression_metrics(actuals, predicted):
    predicted = np.nan_to_num(predicted)
    mae = mean_absolute_error(actuals, predicted)
    mae_asp = (mean_absolute_error(actuals, predicted)/actuals.std())*100
    rmse_asp = (np.sqrt(mean_squared_error(actuals,predicted))/actuals.std())*100
    rmse = print_rmse(actuals, predicted)
    _ = print_mape(actuals, predicted)
    mape = print_mape(actuals, predicted)
    print('    MAE = %0.4f' %mae)
    print("    MAPE = %0.0f%%" %(mape))
    print('    RMSE = %0.4f' %rmse)
    print('    MAE as %% std dev of Actuals = %0.1f%%' %(mae/abs(actuals).std()*100))
    # Normalized RMSE print('RMSE = {:,.Of}'.format(rmse))
    print('    R-Squared (%% ) = %0.0f%%' %(100*r2_score(actuals,predicted)))
    print('    Normalized RMSE (%% of Std Dev of Actuals) = %0.0f%%' %(100*rmse/actuals.std()))
    return mae, mae_asp, rmse_asp
################################################################################
def print_static_rmse(actuals, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    predicted = np.nan_to_num(predicted)
    rmse = np.sqrt(mean_squared_error(actuals[start_from:],predicted[start_from:]))
    std_dev = actuals[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
        print('    R-Squared (%% ) = %0.0f%%' %(100*r2_score(actuals,predicted)))
    return rmse, rmse/std_dev
################################################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(abs(perc_err))
################################################################################
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
def print_classification_header(num_classes, num_labels, target_name):
    ######## This is where you start printing metrics ###############
    if isinstance(num_classes, list) :
        if np.max(num_classes) > 2:
            print('Multi Label (multi-output) Multi-Class Report: %s' %target_name)
            print('#################################################################')
        else:            
            print('Multi Label (multi-output) Binary Class Metrics Report: %s' %target_name)
            print('#################################################################')
    else:
        if num_classes > 2:
            print('Single Label (single-output), Multi-Class Report: %s' %target_name)
            print('#################################################################')
        else:
            print('Single Label, Multi Class Model Metrics Report: %s' %target_name)
            print('#################################################################')

def print_classification_metrics(y_test, y_probs, proba_flag=True):
    """
    #######  Send in the actual_values and prediction_probabilities for binary classes
    This will return back metrics and print them all in a neat format
    """
    y_test = copy.deepcopy(y_test)
    multi_label_flag = False
    multi_class_flag = False
    #### for some cases, you won't get proba, so check the proba_flag
    if proba_flag:
        y_preds = y_probs.argmax(axis=1)
    else:
        y_preds = copy.deepcopy(y_probs)
    ##### check if it is multi-class #####
    if len(np.unique(y_test)) > 2 or max(np.unique(y_test)) >= 2:
        multi_class_flag = True
    elif len(np.unique(y_preds)) > 2 or max(np.unique(y_preds)) >= 2:
        multi_class_flag = True
    ###########  This is where we print the metrics ###################
    try:
        if not multi_class_flag  and not multi_label_flag:
            # Calculate comparison metrics for Binary classification results.
            accuracy = metrics.accuracy_score(y_test, y_preds)
            balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_preds)
            precision = metrics.precision_score(y_test, y_preds)
            f1_score = metrics.f1_score(y_test, y_preds)
            recall = metrics.recall_score(y_test, y_preds)
            if type(np.mean((y_test==y_preds))) == pd.Series:
                print('    Accuracy          = %0.1f%%' %(np.mean(accuracy)*100))
            else:
                print('    Accuracy          = %0.1f%%' %(accuracy*100))
            print('    Balanced Accuracy = %0.1f%%' %(balanced_accuracy*100))
            print('    Precision         = %0.1f%%' %(precision*100))
            if proba_flag:
                average_precision = np.mean(metrics.precision_score(y_test, y_preds, average=None))
            else:
                average_precision = metrics.precision_score(y_test, y_preds, average='macro')
            print('    Average Precision = %0.1f%%' %(average_precision*100))
            print('    Recall            = %0.1f%%' %(recall*100))
            print('    F1 Score          = %0.1f%%' %(f1_score*100))
            if proba_flag:
                roc_auc = metrics.roc_auc_score(y_test, y_probs[:,1])
                #fpr, tpr, threshold = metrics.roc_curve(y_test, y_probs[:,1])
                #roc_auc = metrics.auc(fpr, tpr)
                print('    ROC AUC           = %0.1f%%' %(roc_auc*100))
            else:
                roc_auc = 0
            print('#####################################################')
            return [accuracy, balanced_accuracy, precision, average_precision, f1_score, recall, roc_auc]
        else:
            # Calculate comparison metrics for Multi-Class classification results.
            accuracy = np.mean((y_test==y_preds))
            if multi_label_flag:
                balanced_accuracy = np.mean(metrics.recall_score(y_test, y_preds, average=None))
                precision = metrics.precision_score(y_test, y_preds, average=None)
                average_precision = metrics.precision_score(y_test, y_preds, average='macro')
                f1_score = metrics.f1_score(y_test, y_preds, average=None)
                recall = metrics.recall_score(y_test, y_preds, average=None)
            else:
                balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_preds)
                precision = metrics.precision_score(y_test, y_preds, average = None)
                average_precision = metrics.precision_score(y_test, y_preds,average='macro')
                f1_score = metrics.f1_score(y_test, y_preds, average = None)
                recall = metrics.recall_score(y_test, y_preds, average = None)
            if type(np.mean((y_test==y_preds))) == pd.Series:
                print('    Accuracy          = %0.1f%%' %(np.mean(accuracy)*100))
            else:
                print('    Accuracy          = %0.1f%%' %(accuracy*100))
            print('    Balanced Accuracy (average recall) = %0.1f%%' %(balanced_accuracy*100))
            print('    Average Precision (macro) = %0.1f%%' %(average_precision*100))
            ### these are basically one for each class #####
            print('    Precisions by class:')
            for precisions in precision:
                print('    %0.1f%%  ' %(precisions*100),end="")
            print('\n    Recall Scores by class:')
            for recalls in recall:
                print('    %0.1f%%  ' %(recalls*100), end="")
            print('\n    F1 Scores by class:')
            for f1_scores in f1_score:
                print('    %0.1f%%  ' %(f1_scores*100),end="")
            # Return list of metrics to be added to a Dataframe to compare models.
    except:
        print('    print classification metrics erroring. Continuing...')
    print('\n#####################################################')
    return [accuracy, balanced_accuracy, precision, average_precision, f1_score, recall, 0]
##################################################################################################
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
#####################################################################
#####     REGRESSION CHARTS AND METRICS ARE PRINTED PLOTTED HERE
#####################################################################
import time
from itertools import cycle
def plot_regression_scatters(df, df2, num_vars, kind='scatter', plot_name=''):
    """
    Great way to plot continuous variables fast. Just sent them in and it will take care of the rest!
    """
    figsize = (10, 10)
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
    num_vars_len = len(num_vars)
    col = 2
    start_time = time.time()
    row = len(num_vars)
    fig, ax = plt.subplots(row, col)
    if col < 2:
        fig.set_size_inches(min(15,8),row*5)
        fig.subplots_adjust(hspace=0.5) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    else:
        fig.set_size_inches(min(col*10,20),row*5)
        fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
        fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
    fig.suptitle('Regression Metrics Plots for %s Model' %plot_name, fontsize=20)
    counter = 0
    if row == 1:
        ax = ax.reshape(-1,1).T
    for k in np.arange(row):
        row_color = next(colors)
        for l in np.arange(col):
            try:
                if col==1:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:,k]
                        y = df2[:,k]
                    ax1 = ax[k][l]
                    lineStart = x.min()
                    lineEnd = x.max()
                    ax1.scatter(x, y, color=row_color)
                    ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=next(colors))
                    ax1.set_xlabel('Actuals')
                    ax1.set_ylabel('Predicted')
                    ax1.set_title('Predicted vs Actuals Plot for Target = %s' %num_vars[k])
                else:
                    if row == 1:
                        x = df[:]
                        y = df2[:]
                    else:
                        x = df[:,k]
                        y = df2[:,k]
                    lineStart = x.min()
                    lineEnd = x.max()
                    if l == 0:
                        ax1 = ax[k][l]
                        ax1.scatter(x, y,  color = row_color)
                        ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = next(colors))
                        ax1.set_xlabel('Actuals')
                        ax1.set_ylabel('Predicted')
                        ax1.set_title('Predicted vs Actuals Plot for Target = %s' %num_vars[k])
                    else:
                        ax1 = ax[k][l]
                        try:
                            assert y.shape[1]
                            ax1.hist((x-y.ravel()), density=True,color = row_color)
                        except:
                            ax1.hist((x-y), density=True,color = row_color)
                        ax1.axvline(linewidth=2, color='k')
                        ax1.set_title('Residuals Plot for Target = %s' %num_vars[k])
            except:
                if col == 1:
                    counter += 1
                else:
                    ax[k][l].set_title('No Predicted vs Actuals Plot for plot as %s is not numeric' %num_vars[k])
                    counter += 1
    print('Regression Plots completed in %0.3f seconds' %(time.time()-start_time))
################################################################################
def plot_regression_residuals(y_test, y_test_preds, target, project_name, num_labels):
    """
    Another set of plots for continuous variables.
    """
    try:
        if isinstance(target, str):
            colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
            row_color = next(colors)
            plt.figure(figsize=(15,6))
            ax1 = plt.subplot(1, 2, 1)
            residual = pd.Series((y_test - y_test_preds))
            residual.plot(ax=ax1, color='b')
            ax1.set_title('Residuals by each row in held-out set')
            ax1.axhline(y=0.0, linewidth=2, color=next(colors))
            pdf = save_valid_predictions(y_test, y_test_preds.ravel(), project_name, num_labels)
            ax2 = plt.subplot(1, 2, 2)
            pdf.plot(ax=ax2)
            ax2.set_title('Actuals vs Predictions by each row in held-out set')
        else:
            pdf = save_valid_predictions(y_test, y_test_preds, project_name, num_labels)
            plt.figure(figsize=(15,6))
            colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
            for i in range(num_labels):
                row_color = next(colors)
                ax1 = plt.subplot(1, num_labels, i+1)
                residual = pd.Series((y_test[:,i] - y_test_preds[:,i]))
                residual.plot(ax=ax1, color=row_color)
                ax1.set_title(f"Actuals_{i} (x-axis) vs. Residuals_{i} (y-axis)")
                ax1.axhline(y=0.0, linewidth=2, color=next(colors))
            plt.figure(figsize=(15, 6)) 
            colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgk')
            for j in range(num_labels):
                row_color = next(colors)
                pair_cols = ['actuals_'+str(j), 'predictions_'+str(j)]
                ax2 = plt.subplot(1, num_labels, j+1)
                pdf[pair_cols].plot(ax=ax2)
                ax2.set_title('Actuals_{j} vs Predictions_{j} for each row ')
    except:
        print('Regression plots erroring. Continuing...')
#############################################################################################
import os
def save_valid_predictions(y_test, y_preds, project_name, num_labels):
    if num_labels == 1:
        pdf = pd.DataFrame([y_test, y_preds])
        pdf = pdf.T
        pdf.columns= ['actuals','predictions']
    else:
        pdf = pd.DataFrame(np.c_[y_test, y_preds])
        act_names = ['actuals_'+str(x) for x in range(y_test.shape[1])]
        pred_names = ['predictions_'+str(x) for x in range(y_preds.shape[1])]
        pdf.columns = act_names + pred_names
    preds_file = project_name+'_predictions.csv'
    preds_path = os.path.join(project_name, preds_file)
    pdf.to_csv(preds_path,index=False)
    print('Saved predictions in %s file' %preds_path)
    return pdf
#########################################################################################    
import matplotlib.pyplot as plt
from IPython.display import Image, display
def print_one_image_from_dataset(train_ds, classes):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[labels[i]])
        plt.axis("off");
#########################################################################################
def predict_plot_images(model, test_ds, classes):
    plt.figure(figsize=(10, 10))
    for images, labels in test_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        plt.tight_layout()
        
        img = tf.keras.preprocessing.image.img_to_array(images[i])                    
        img = np.expand_dims(img, axis=0)  

        pred=model.predict(img)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("Actual Label: %s" % classes[labels[i]])
        plt.text(1, 240, "Predicted Label: %s" % classes[np.argmax(pred)], fontsize=12)

        plt.axis("off")
#######################################################################################
def get_model_defaults(keras_options, model_options, targets):
    num_classes = model_options["num_classes"]
    num_labels = model_options["num_labels"]
    modeltype = model_options["modeltype"]
    patience = check_keras_options(keras_options, "patience", 10)
    use_bias = check_keras_options(keras_options, 'use_bias', True)
    optimizer = check_keras_options(keras_options,'optimizer', Adam(lr=0.01, beta_1=0.9, beta_2=0.999))
    if modeltype == 'Regression':
        reg_loss = check_keras_options(keras_options,'loss','mae') ### you can use tf.keras.losses.Huber() instead
        #val_metrics = [check_keras_options(keras_options,'metrics',keras.metrics.RootMeanSquaredError(name='rmse'))]
        METRICS = [keras.metrics.MeanAbsoluteError(name='mae'), keras.metrics.RootMeanSquaredError(name='rmse'),]
        val_metrics = check_keras_options(keras_options,'metrics',METRICS)
        num_predicts = 1*num_labels
        if num_labels <= 1:
            val_loss = check_keras_options(keras_options,'loss', reg_loss)
        else:
            val_loss = []
            for i in range(num_labels):
                val_loss.append(reg_loss)
        ####### If you change the val_metrics above, you must also change its name here ####
        val_metric = 'rmse'
        output_activation = 'linear' ### use "relu" or "softplus" if you want positive values as output
    elif modeltype == 'Classification':
        ##### This is for Binary Classification Problems
        #val_loss = check_keras_options(keras_options,'loss','sparse_categorical_crossentropy')
        #val_metrics = [check_keras_options(keras_options,'metrics','AUC')]
        #val_metrics = check_keras_options(keras_options,'metrics','accuracy')
        cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        val_loss = check_keras_options(keras_options,'loss', cat_loss)
        bal_acc = BalancedSparseCategoricalAccuracy()
        val_metrics = check_keras_options(keras_options,'metrics',bal_acc)
        if num_labels <= 1:
            num_predicts = int(num_classes*num_labels)
        else:
            #### This is for multi-label problems wihere number of classes will be a list
            num_predicts = num_classes
        output_activation = "sigmoid"
        ####### If you change the val_metrics above, you must also change its name here ####
        val_metric = 'balanced_sparse_categorical_accuracy'
    else:
        #### this is for multi-class problems ####
        cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #val_loss = check_keras_options(keras_options,'loss','sparse_categorical_crossentropy')
        #val_metrics = check_keras_options(keras_options,'metrics','accuracy')
        if num_labels <= 1:
            num_predicts = int(num_classes*num_labels)
            val_loss = check_keras_options(keras_options,'loss', cat_loss)
            bal_acc = BalancedSparseCategoricalAccuracy()
            val_metrics = check_keras_options(keras_options, 'metrics', bal_acc)
        else:
            #### This is for multi-label problems wihere number of classes will be a list
            num_predicts = num_classes
            val_loss = []
            for i in range(num_labels):
                val_loss.append(cat_loss)
            bal_acc = BalancedSparseCategoricalAccuracy()
            val_metrics = check_keras_options(keras_options, 'metrics', bal_acc)
        output_activation = 'softmax'
        ####### If you change the val_metrics above, you must also change its name here ####
        val_metric = 'balanced_sparse_categorical_accuracy'
    ##############  Suggested number of neurons in each layer ##################
    if modeltype == 'Regression':
        val_monitor = check_keras_options(keras_options, 'monitor', 'val_'+val_metric)
        val_mode = check_keras_options(keras_options,'mode', 'min')
    elif modeltype == 'Classification':
        ##### This is for Binary Classification Problems
        if num_labels <= 1:
            val_monitor = check_keras_options(keras_options,'monitor', 'val_'+val_metric)
            val_mode = check_keras_options(keras_options,'mode', 'max')
        else:
            val_metric = 'balanced_sparse_categorical_accuracy'
            ### you cannot combine multiple metrics here unless you write a new function.
            target_A = targets[0]
            val_monitor = 'val_loss' ### this combines all losses and is best to minimize
            val_mode = check_keras_options(keras_options,'mode', 'min')
        #val_monitor = check_keras_options(keras_options,'monitor', 'val_auc')
        #val_monitor = check_keras_options(keras_options,'monitor', 'val_accuracy')
    else:
        #### this is for multi-class problems
        if num_labels <= 1:
            val_monitor = check_keras_options(keras_options,'monitor', 'val_'+val_metric)
            val_mode = check_keras_options(keras_options, 'mode', 'max')
        else:
            val_metric = 'balanced_sparse_categorical_accuracy'
            ### you cannot combine multiple metrics here unless you write a new function.
            target_A = targets[0]
            val_monitor = 'val_loss'
            val_mode = check_keras_options(keras_options,'mode', 'min')
        #val_monitor = check_keras_options(keras_options, 'monitor','val_accuracy')
    ##############################################################################
    keras_options["mode"] = val_mode
    keras_options["monitor"] = val_monitor
    keras_options["metrics"] = val_metrics
    keras_options['loss'] = val_loss
    keras_options["patience"] = patience
    keras_options['use_bias'] = use_bias
    keras_options['optimizer'] = optimizer
    return keras_options, model_options, num_predicts, output_activation
###############################################################################
def get_uncompiled_model(inputs, result, output_activation, 
                    num_predicts, modeltype, cols_len, targets):
    ### The next 3 steps are most important! Don't mess with them! 
    #model_preprocessing = Model(inputs, meta_outputs)
    #preprocessed_inputs = model_preprocessing(inputs)
    #result = model_body(preprocessed_inputs)
    ##### now you can add the final layer here #########
    multi_label_predictions = defaultdict(list)
    if isinstance(num_predicts, int):
        key = 'predictions'
        if modeltype == 'Regression':
            ### this will be just 1 in regression ####
            if num_predicts > 1:
                for each_label in range(num_predicts):
                    value = layers.Dense(1, activation=output_activation,
                            name=targets[each_label])(result)
                    multi_label_predictions[key].append(value)
            else:
                value = layers.Dense(1, activation=output_activation,
                        name=targets[0])(result)
                multi_label_predictions[key].append(value)
        else:
            ### this will be number of classes in classification ###
            value = layers.Dense(num_predicts, activation=output_activation,
                                name=targets[0])(result)
            multi_label_predictions[key].append(value)
    else:
        #### This will be for multi-label, multi-class predictions only ###
        for each_label in range(len(num_predicts)):
            key = 'predictions'
            if modeltype == 'Regression':
                ### this will be just 1 in regression ####
                value = layers.Dense(1, activation=output_activation,
                        name=targets[0])(result)
            else:
                ### this will be number of classes in classification ###
                value = layers.Dense(num_predicts[each_label], activation=output_activation,
                                    name=targets[each_label])(result)
            multi_label_predictions[key].append(value)
    outputs = multi_label_predictions[key] ### outputs will be a list of Dense layers
    ##### Set the inputs and outputs of the model here
    
    uncompiled_model = Model(inputs=inputs, outputs=outputs)
    return uncompiled_model

#####################################################################################
def get_compiled_model(inputs, meta_outputs, output_activation, num_predicts, modeltype, 
                       optimizer, val_loss, val_metrics, cols_len, targets):
    model = get_uncompiled_model(inputs, meta_outputs, output_activation, 
                        num_predicts, modeltype, cols_len, targets)
    model.compile(
        optimizer=optimizer,
        loss=val_loss,
        metrics=val_metrics,
    )
    return model
###############################################################################
def add_inputs_outputs_to_auto_model_body(model_body, inputs, meta_outputs, nlp_flag=False):
    """
    This is specially for "auto" model types only. It requires special handling.
    """
    if nlp_flag:
        wide, deep, nlp_outputs = meta_outputs
    else:
        wide, deep = meta_outputs
    ##### This is the simplest way to convert a sequential model to functional!
    for num, each_layer in enumerate(model_body.layers):
        if num == 0:
            final_outputs = each_layer(deep)
        else:
            final_outputs = each_layer(final_outputs)
    model_body = layers.concatenate([wide, final_outputs], name='auto_concatenate_layer')
    return model_body
#################################################################################
def add_inputs_outputs_to_model_body(model_body, inputs, meta_outputs):
    ##### This is the simplest way to convert a sequential model to functional!
    for num, each_layer in enumerate(model_body.layers):
        if num == 0:
            final_outputs = each_layer(meta_outputs)
        else:
            final_outputs = each_layer(final_outputs)
    return final_outputs
###############################################################################
class BalancedSparseCategoricalAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_sparse_categorical_accuracy', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)
#####################################################################################
def check_keras_options(keras_options, name, default):
    try:
        if keras_options[name]:
            value = keras_options[name] 
        else:
            value = default
    except:
        value = default
    return value
#####################################################################################
# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      self.model.optimizer.lr.numpy()))
#####################################################################################
# Function for decaying the learning rate.
# You can define any decay function you need.
def schedules(epoch, lr):
    return lr * (0.997 ** np.floor(epoch / 2))

LEARNING_RATE = 0.01
def decay(epoch):
    return LEARNING_RATE - 0.0099 * (0.75 ** (1 + epoch/2))
#######################################################################################
import os
def get_callbacks(val_mode, val_monitor, patience, learning_rate, save_weights_only, steps=100):
    """
    ####################################################################################
    #####    LEARNING RATE SCHEDULING : Setup Learning Rate Multiple Ways #########
    ####################################################################################
    """
    keras.backend.clear_session()
    logdir = "deep_autoviml"
    callbacks_dict = {}
    tensorboard_logpath = os.path.join(logdir,"mylogs")
    print('Tensorboard log directory can be found at: %s' %tensorboard_logpath)

    cp = keras.callbacks.ModelCheckpoint("deep_autoviml", save_best_only=True,
                                         save_weights_only=save_weights_only, save_format='tf')
    ### sometimes a model falters and restore_best_weights gives len() not found error. So avoid True option!
    lr_patience = max(5,int(patience*0.5))
    rlr = keras.callbacks.ReduceLROnPlateau(monitor=val_monitor, factor=0.90,
                    patience=lr_patience, min_lr=1e-6, mode='auto', min_delta=0.00001, cooldown=0, verbose=1)

    ###### This is one variation of onecycle #####
    onecycle = OneCycleScheduler(steps=steps, lr_max=0.1)

    ###### This is another variation of onecycle #####
    onecycle2 = OneCycleScheduler2(iterations=steps, max_rate=0.05)

    lr_sched = keras.callbacks.LearningRateScheduler(schedules)

      # Setup Learning Rate decay.
    lr_decay_cb = keras.callbacks.LearningRateScheduler(decay)

    es = keras.callbacks.EarlyStopping(monitor=val_monitor, min_delta=0.00001, patience=patience,
                        verbose=1, mode=val_mode, baseline=None, restore_best_weights=True)
    
    tb = keras.callbacks.TensorBoard(log_dir=tensorboard_logpath,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1
                         )

    pr = PrintLR()

    callbacks_dict['onecycle'] = onecycle
    callbacks_dict['onecycle2'] = onecycle2
    callbacks_dict['check_point'] = cp
    callbacks_dict['scheduler'] = lr_sched
    callbacks_dict['early_stop'] = es
    callbacks_dict['tensor_board'] = tb
    callbacks_dict['print'] = pr
    callbacks_dict['reducer'] = rlr
    callbacks_dict['decay'] = lr_decay_cb

    return callbacks_dict, tensorboard_logpath
####################################################################################
def get_chosen_callback(callbacks_dict, keras_options):
    ##### Setup Learning Rate decay.
    #### Onecycle is another fast way to find the best learning in large datasets ######
    ##########  This is where we look for various schedulers
    if keras_options['lr_scheduler'] == "onecycle2":
        lr_scheduler = callbacks_dict['onecycle2']
    elif keras_options['lr_scheduler'] == 'onecycle':
        lr_scheduler = callbacks_dict['onecycle']
    elif keras_options['lr_scheduler'] == 'reducer':
        lr_scheduler = callbacks_dict['reducer']
    elif keras_options['lr_scheduler'] == 'decay':
        lr_scheduler = callbacks_dict['decay']
    elif keras_options['lr_scheduler'] == "scheduler":
        lr_scheduler = callbacks_dict['scheduler']
    else:
        lr_scheduler = callbacks_dict['scheduler']
    return lr_scheduler
################################################################################################
def get_chosen_callback2(callbacks_dict, keras_options):
    ##### Setup Learning Rate decay.
    #### Onecycle is another fast way to find the best learning in large datasets ######
    ##########  This is where we look for various schedulers
    if keras_options['lr_scheduler'] == "onecycle2":
        lr_scheduler = callbacks_dict['onecycle2']
    elif keras_options['lr_scheduler'] == 'onecycle':
        lr_scheduler = callbacks_dict['onecycle']
    elif keras_options['lr_scheduler'] == 'rlr':
        lr_scheduler = callbacks_dict['rlr']
    elif keras_options['lr_scheduler'] == 'decay':
        lr_scheduler = callbacks_dict['lr_decay_cb']
    else:
        lr_scheduler = callbacks_dict['lr_sched']
        keras_options['lr_scheduler'] = "lr_sched"
    return lr_scheduler
################################################################################################
import math
class OneCycleScheduler2(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        if rate < 0:
            rate = self.start_rate
            self.iteration = 0
            K.clear_session()
        K.set_value(self.model.optimizer.lr, rate)
#####################################################################################################
def find_columns_with_infinity(df):
    """
    This function finds all columns in a dataframe that have inifinite values (np.inf or -np.inf)
    It returns a list of column names. If the list is empty, it means no columns were found.
    """
    add_cols = []
    sum_cols = 0
    for col in df.columns:
        inf_sum1 = 0 
        inf_sum2 = 0
        inf_sum1 = len(df[df[col]==np.inf])
        inf_sum2 = len(df[df[col]==-np.inf])
        if (inf_sum1 > 0) or (inf_sum2 > 0):
            add_cols.append(col)
            sum_cols += inf_sum1
            sum_cols += inf_sum2
    return add_cols
#####################################################################################################
import copy
def drop_rows_with_infinity(df, cols_list, fill_value=None):
    """
    This feature engineering function will fill infinite values in your data with a fill_value.
    You might need this function during deep_learning models where infinite values don't work.
    You can also leave the fill_value as None which means we will drop the rows with infinity.
    This function checks for both negative and positive infinity values to fill or remove.
    """
    # first you must drop rows that have inf in them ####
    print('    Shape of dataset initial: %s' %(df.shape[0]))
    corr_list_copy = copy.deepcopy(cols_list)
    init_rows = df.shape[0]
    if fill_value:
        for col in corr_list_copy:
            ### Capping using the n largest value based on n given in input.
            maxval = df[col].max()  ## what is the maximum value in this column?
            minval = df[col].min()
            if maxval == np.inf:
                sorted_list = sorted(df[col].unique())
                ### find the n_smallest values after the maximum value based on given input n
                next_best_value_index = sorted_list.index(np.inf) - 1
                capped_value = sorted_list[next_best_value_index]
                df.loc[df[col]==maxval, col] =  capped_value ## maximum values are now capped
            if minval == -np.inf:
                sorted_list = sorted(df[col].unique())
                ### find the n_smallest values after the maximum value based on given input n
                next_best_value_index = sorted_list.index(-np.inf)+1
                capped_value = sorted_list[next_best_value_index]
                df.loc[df[col]==minval, col] =  capped_value ## maximum values are now capped
        print('        capped all rows with infinite values in data')
    else:
        for col in corr_list_copy:
            df = df[df[col]!=np.inf]
            df = df[df[col]!=-np.inf]
        dropped_rows = init_rows - df.shape[0]
        print('        dropped %d rows due to infinite values in data' %dropped_rows)
        print('    Shape of dataset after dropping rows: %s' %(df.shape[0]))
    ###  Double check that all columns have been fixed ###############
    cols_with_infinity = find_columns_with_infinity(df)
    if cols_with_infinity:
        print('There are still %d columns with infinite values. Returning...' %len(cols_with_infinity))
    else:
        print('There are no columns with infinite values.')
    return df
################################################################################################
def get_hidden_layers(data_dim):
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
    return dense_layer1, dense_layer2, dense_layer3
###################################################################################################