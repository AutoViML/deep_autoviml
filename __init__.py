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

from .deep_autoviml import run
from .modeling.predict_model import load_test_data, predict_model
################################################################################
if __name__ == "__main__":
    module_type = 'Running'
else:
    module_type = 'Imported'
version_number = __version__
print("""
%s deep_auto_viml version=%s Build deep learning models, pipelines, fast!
--- 
model, dictionary = deep_auto_viml.run(train_data_or_file, target, keras_model_type)
---
model.save_to_file_or_cloud(model_filename, gcp_project_id, gcp_bucket...)
---
model = load_from_file_or_cloud(model_filename, gcp_project_id, gcp_bucket...)
model.predict(test_data)
                                """ %(module_type, version_number))
################################################################################
