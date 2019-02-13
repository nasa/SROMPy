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

import pytest
import numpy as np
import os
import sys

if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('.')

    sys.path.insert(0, base_path)


from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector
from SROMPy.target import RandomVector


def test_invalid_init_parameter_values_rejected():

    # Ensure exception for invalid srom dimensions.
    with pytest.raises(ValueError):
        SROM(0, 1)
        SROM(-1, 1)
        SROM(10, 0)
        SROM(10, 3)


def test_invalid_optimize_parameter_values_rejected():

    pass
