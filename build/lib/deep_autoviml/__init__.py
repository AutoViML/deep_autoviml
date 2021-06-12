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
# -*- coding: utf-8 -*-
################################################################################
#     deep_auto_viml - build and test multiple Tensorflow 2.0 models and pipelines
#     Python v3.6+ tensorflow v2.4.1+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .__version__ import __version__
__all__ = ['data_load', 'models', 'modeling', 'preprocessing', 'utilities']
import pdb

from .deep_autoviml import fit
from .modeling.predict_model import load_test_data, predict
from deep_autoviml.modeling.predict_model import predict_images
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""
%s deep_auto_viml. version=%s
from deep_autoviml import deep_autoviml as deepauto
-------------------
model, cat_vocab_dict = deepauto.fit(train, target, keras_model_type="auto",
		project_name="deep_autoviml", keras_options={}, model_options={}, 
		save_model_flag=True, use_my_model='', model_use_case='', verbose=0)

predictions = deepauto.predict(model, project_name, test_dataset=test,
                                 keras_model_type=keras_model_type, 
                                 cat_vocab_dict=cat_vocab_dict)
                                """ %(module_type, version_number))
################################################################################
