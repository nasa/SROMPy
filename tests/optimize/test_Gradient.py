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
from SROMPy.optimize import Gradient
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
def gradient(sample_random_vector, valid_srom):
    return Gradient(srom=valid_srom,
                    target_random_variable=sample_random_vector,
                    error='mean')


@pytest.fixture
def gradient_joint(sample_random_vector, valid_srom):
    return Gradient(srom=valid_srom,
                    target_random_variable=sample_random_vector,
                    error='SSE',
                    scale=0.01,
                    joint_opt=True)


def test_invalid_init_parameter_values_rejected(sample_random_vector, valid_srom):
    # Ensure no exception using default parameters.
    Gradient(srom=valid_srom, target_random_variable=sample_random_vector)

    # Ensure exception if obj_weights parameter is too small.
    with pytest.raises(ValueError):
        Gradient(srom=valid_srom, target_random_variable=sample_random_vector,
                 obj_weights=np.ones((2,)))

    # Ensure exception if specified error parameter is invalid.
    with pytest.raises(ValueError):
        Gradient(srom=valid_srom, target_random_variable=sample_random_vector,
                 error='test')


def test_evaluate_returns_expected_result(valid_srom, gradient):
    samples = np.ones((valid_srom._size, valid_srom._dim))
    probabilities = np.ones(valid_srom._size)

    results = gradient.evaluate(samples, probabilities)

    assert isinstance(results, np.ndarray)
    assert results.size == valid_srom._size


def test_gradient_joint(valid_srom, gradient_joint):
    samples = np.ones((valid_srom.size, valid_srom.dim))
    probabilities = np.ones(valid_srom.size)

    results = gradient_joint.evaluate(samples, probabilities)

    assert isinstance(results, np.ndarray)
    assert results.size == (valid_srom.size * valid_srom.dim + valid_srom.size)
