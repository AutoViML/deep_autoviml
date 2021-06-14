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
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep_autoviml",
    version="0.0.40",
    author="Ram Seshadri",
    # author_email="author@example.com",
    description="Automatically Build Deep Learning Models and Pipelines fast!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    url="https://github.com/AutoViML/deep_autoviml",
    packages = [
        "deep_autoviml",
        "deep_autoviml.data_load",
        "deep_autoviml.modeling",
        "deep_autoviml.models",
        "deep_autoviml.preprocessing",
        "deep_autoviml.utilities",
    ],
    include_package_data=True,
    install_requires=[
        "ipython",
        "jupyter",
        "tensorflow==2.5.0",
        "pandas",
        "matplotlib",
        "numpy==1.19.2",
        "scikit-learn>=0.23.1",
        "regex",
        "emoji",
        "storm-tuner",
        "optuna",
        "tensorflow_hub==0.12.0",
        "tensorflow-text==2.5.0",
        "xlrd"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
