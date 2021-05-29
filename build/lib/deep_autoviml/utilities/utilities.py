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
from tensorflow.keras.models import Model, load_model
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
            for gpu in logical_gpus:
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
def get_uncompiled_model(inputs, result, model_body, output_activation, 
                    num_predicts, num_labels, cols_len):
    ### The next 3 steps are most important! Don't mess with them! 
    #model_preprocessing = Model(inputs, meta_outputs)
    #preprocessed_inputs = model_preprocessing(inputs)
    #result = model_body(preprocessed_inputs)
    ##### now you
    multi_label_predictions = defaultdict(list)
    for each_label in range(num_labels):
        key = 'predictions'        
        value = layers.Dense(num_predicts, activation=output_activation,
                            name='output_'+str(each_label))(result)
        multi_label_predictions[key].append(value)
    outputs = multi_label_predictions[key] ### outputs will be a list of Dense layers

    ##### Set the inputs and outputs of the model here
    uncompiled_model = Model(inputs=inputs, outputs=outputs)
    return uncompiled_model

def get_compiled_model(inputs, meta_outputs, output_activation, num_predicts, num_labels, 
                      model_body, optimizer, val_loss, val_metrics, cols_len):
    model = get_uncompiled_model(inputs, meta_outputs, model_body, output_activation, 
                        num_predicts, num_labels, cols_len)
    model.compile(
        optimizer=optimizer,
        loss=val_loss,
        metrics=val_metrics,
    )
    return model
###############################################################################
def add_inputs_outputs_to_model_body(model_body, inputs, meta_outputs):
    ##### This is the simplest way to convert a sequential model to functional!
    for num, each_layer in enumerate(model_body.layers):
        if num == 0:
            final_outputs = each_layer(meta_outputs)
        else:
            final_outputs = each_layer(final_outputs)
    return final_outputs

###############################################################################
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
    try:
        if isinstance(test_label, tuple):
            dict_row = list(test_label.as_numpy_iterator())[0]
        else:
            dict_row = test_label
        print("Printing 10 samples from labels data:")
        preds = list(dict_row.element_spec[0].keys())
        if dict_row.element_spec[0][preds[0]].shape[0] is None or isinstance(
                dict_row.element_spec[0][preds[0]].shape[0], int):
            for feats, labs in dict_row.take(1): 
                print(labs[:10])
    except:
        print(list(test_label.as_numpy_iterator())[0])
###########################################################################################
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
def plot_history(history, metric, num_labels):
    if num_labels == 1:
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
        ### You must choose one of the label outputs to monitor - we will choose the last one
        for i in range(num_labels):
            fig = plt.figure(figsize=(15,6))
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title('Model Training vs Validation Loss')
            plot_one_history_metric(history, "loss", ax1)
            ax2 = plt.subplot(1, 2, 2)
            metric2 = 'output_'+str(i)+'_' + metric
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

def print_classification_model_stats(y_true, predicted):
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
def plot_classification_results(y_true, y_probas, labels, target_names):
    try:
        y_pred = y_probas.argmax(axis=1)
        fig, axes = plt.subplots(1,2,figsize=(15,6))
        draw_confusion_matrix(y_true, y_pred, 'Confusion Matrix', ax=axes[0])
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
def draw_confusion_matrix(y_test,y_pred, model_name='Model',ax=''):
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
                         index = np.unique(y_test).tolist(),
                         columns = np.unique(y_test).tolist(),
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
                predicted_x = predicted[:]
            print('Regression Metrics for Target=%s' %cols[i])
            mae, mae_asp, rmse_asp = print_regression_metrics(actuals_x, predicted_x)
    else:
        mae, mae_asp, rmse_asp = print_regression_metrics(actuals, predicted)
    return mae, mae_asp, rmse_asp
################################################################################
def print_regression_metrics(actuals, predicted):
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
    print('    Normalized RMSE (%% of MinMax of Actuals) = %0.0f%%' %(100*rmse/abs(actuals.max()-actuals.min())))
    print('    Normalized RMSE (%% of Std Dev of Actuals) = %0.0f%%' %(100*rmse/actuals.std()))
    return mae, mae_asp, rmse_asp
################################################################################
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
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
import pdb
def print_classification_metrics(y_test, y_probs, proba_flag=True):
    """
    #######  Send in the actual_values and prediction_probabilities for binary classes
    This will return back metrics and print them all in a neat format
    """
    y_test = copy.deepcopy(y_test)
    multi_label_flag = False
    multi_class_flag = False
    try:
        if y_test.shape[1] > 0:
            multi_label_flag = True
    except:
        pass
    ######## This is where you start printing metrics ###############
    if len(np.unique(y_test)) > 2:
        multi_class_flag = True
        print('Multi Class Model Metrics Report')
        print('#####################################################')
    else:
        print('Binary Class Model Metrics Report')
        print('#####################################################')
    #### for some cases, you won't get proba, so check the proba_flag
    if proba_flag:
        if multi_label_flag:
            y_preds = (y_probs>0.5).astype(int)
        else:
            y_preds = y_probs.argmax(axis=1)
    else:
        y_preds = copy.deepcopy(y_probs)
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
            fpr, tpr, threshold = metrics.roc_curve(y_test, y_probs[:,1])
            roc_auc = metrics.auc(fpr, tpr)
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
                    ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color=row_color)
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
                        row_color = next(colors)
                        ax1.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = row_color)
                        ax1.set_xlabel('Actuals')
                        ax1.set_ylabel('Predicted')
                        ax1.set_title('Predicted vs Actuals Plot for Target = %s' %num_vars[k])
                    else:
                        ax1 = ax[k][l]
                        row_color = next(colors)
                        ax1.hist((x-y), density=True,color = row_color)
                        ax1.axvline(color='k')
                        ax1.set_title('Residuals Plot for Target = %s' %num_vars[k])
            except:
                if col == 1:
                    counter += 1
                else:
                    ax[k][l].set_title('No Predicted vs Actuals Plot for plot as %s is not numeric' %num_vars[k])
                    counter += 1
    print('Regression Plots completed in %0.3f seconds' %(time.time()-start_time))
################################################################################
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

