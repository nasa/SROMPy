# Copyright 2018 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.

# The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed
# under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SROMPy",
    version="0.1.0",
    author="James Warner ",
    author_email="james.e.warner@nasa.gov",
    description="Stochastic Reduced Order Models with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA/SROMPy",
    packages=["SROMPy",
              "SROMPy.optimize",
              "SROMPy.postprocess",
              "SROMPy.srom",
              "SROMPy.target"],
    package_dir={'SROMPy': 'SROMPy'},
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
