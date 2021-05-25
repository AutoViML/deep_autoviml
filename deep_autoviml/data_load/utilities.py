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
#############################################################################################
def print_one_row_from_tf_dataset(test_ds, preds):
    """
    No matter how big a dataset or batch size, this handy function will print the first row.
    This way you can test what's in each row of a tensorflow dataset that you sent in as input
    You need to provide at least one column in the dataset for it to check if it should print it.
    Inputs:
    -------
    test_ds: tf.data.DataSet - this must be batched and num_epochs must be an integer.
                - otherwise it won't print!
    preds: list - give the name of at least one column in the dataset in list format []
    """
    try:
        if isinstance(test_ds, tuple):
            dict_row = list(test_ds.as_numpy_iterator())[0]
        else:
            dict_row = test_ds
        print("\nPrinting one row from the dataset:")
        if dict_row.element_spec[0][preds[0]].shape[0] is None:
            for batch, head in dict_row.take(1):
                for labels, value in batch.items():
                    print("{:40s}: {}".format(labels, value.numpy()[:4]))
    except:
        print(list(test_ds.as_numpy_iterator())[0])
#########################################################################################
def print_one_row_from_tf_label(test_label, preds):
    """
    No matter how big a dataset or batch size, this handy function will print the first row.
    This way you can test what's in each row of a tensorflow dataset that you sent in as input
    You need to provide at least one column in the dataset for it to check if it should print it.
    Inputs:
    -------
    test_label: tf.data.DataSet - this must be batched and num_epochs must be an integer.
                - otherwise it won't print!
    preds: list - give the name of at least one column in the dataset in list format []
    """
    try:
        if isinstance(test_label, tuple):
            dict_row = list(test_label.as_numpy_iterator())[0]
        else:
            dict_row = test_label
        print("\nPrinting 10 samples from labels data:")
        if dict_row.element_spec[0][preds[0]].shape[0] is None:
            for feats, labs in dict_row.take(1): 
                print(labs[:10])
    except:
        print(list(test_label.as_numpy_iterator())[0])
###########################################################################################
