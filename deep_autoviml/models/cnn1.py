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
import tensorflow as tf
from tensorflow import keras
#### Make sure it is Tensorflow 2.4 or greater!
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Embedding, Reshape, Dropout, Dense
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout, Conv1D
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
############################################################################################
model = tf.keras.Sequential()
model.add(Reshape((-1, 1)))  ### you need to make input as 3-D for CNN models
#model.add(Conv1D(100, 32, name='conv1', padding="same", activation="relu", strides=2,  data_format='channels_first'))
model.add(Conv1D(100, 32, name='conv1', padding="same", activation="relu", strides=2,  data_format='channels_last'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.5))
model.add(Reshape((-1, 1)))  ### you need to make input as 3-D for CNN models
#model.add(Conv1D(64, 16, name='conv2', padding="same", activation="relu", strides=2,  data_format='channels_first'))
model.add(Conv1D(64, 16, name='conv2', padding="same", activation="relu", strides=2,  data_format='channels_last'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dropout(0.25))

