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

import numpy as np
import pytest
import os
import sys

if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('.')

    sys.path.insert(0, base_path)


from SROMPy.srom import SROM
from SROMPy.postprocess import Postprocessor
from SROMPy.target import SampleRandomVector


@pytest.fixture
def sample_random_vector():

    np.random.seed(1)
    random_vector = np.random.rand(10)
    return SampleRandomVector(random_vector)


@pytest.fixture
def valid_srom():
    return SROM(10, 1)


@pytest.fixture
def initialized_srom(valid_srom):
    valid_srom.set_params(np.random.rand(10), np.random.rand(10))
    return valid_srom


# TODO: Find a way to test that Postprocessor checks for required functions
#       on the random entity.
# @pytest.mark.parametrize('required_function', ['compute_cdf',
#                                                'compute_moments'])
# def test_checks_for_random_entity_required_functions(initialized_srom,
#                                                      sample_random_vector,
#                                                      required_function):
#
#     # delatter did not seem to work on a function, so this is a workaround.
#     setattr(sample_random_vector, required_function, "str")
#     delattr(sample_random_vector, required_function)
#
#     with pytest.raises(TypeError):
#         Postprocessor(initialized_srom, sample_random_vector)
