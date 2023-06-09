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


from SROMPy.optimize import Optimizer
from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector


@pytest.fixture
def sample_random_vector():

    np.random.seed(1)
    random_vector = np.random.rand(10)
    return SampleRandomVector(random_vector)


@pytest.fixture
def valid_srom():
    return SROM(10, 1)


def test_invalid_init_parameter_values_rejected(sample_random_vector,
                                                valid_srom):

    # Ensure no exception using default parameters.
    Optimizer(sample_random_vector, valid_srom)

    # TODO: Ensure no exception using valid values for all parameters.
    Optimizer(sample_random_vector,
              valid_srom,
              np.array([1., 1., 1.]))

    # Ensure exception for invalid target parameter.
    with pytest.raises(TypeError):
        Optimizer([], valid_srom)

    with pytest.raises(TypeError):
        Optimizer(np.zeros(10), valid_srom)

    # Ensure exception for invalid srom parameter.
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, None)

    # Ensure exception for invalid weights.
    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, np.array([[1., 1.],
                                                              [1., 1.]]))

    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, np.array([1., 1.]))

    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, np.array([1., 1., -1.]))

    # Ensure invalid error strings are rejected.
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, valid_srom, error=4)

    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, error="BEST")

    # Ensure max moment is a positive integer.
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, valid_srom, max_moment="five")

    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, max_moment=0)

    # Ensure cdf_grid_pts is a positive integer.
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, valid_srom, cdf_grid_pts="five")

    with pytest.raises(ValueError):
        Optimizer(sample_random_vector, valid_srom, cdf_grid_pts=0)

    # Add the scale checker for scale parameter
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, valid_srom, scale="scale")


def test_invalid_get_optimal_params_parameter_values_rejected(sample_random_vector,
                                                              valid_srom):

    # Ensure no errors with valid parameters.
    optimizer = Optimizer(sample_random_vector, valid_srom)

    # Ensure num_test_samples is positive integer.
    with pytest.raises(TypeError):
        optimizer.get_optimal_params(num_test_samples="One")

    with pytest.raises(ValueError):
        optimizer.get_optimal_params(num_test_samples=0)

    # Add the qmc checker
    with pytest.raises(ValueError):
        optimizer.get_optimal_params(qmc_engine="wrong_engine")


def test_get_optimal_params_expected_output(sample_random_vector, valid_srom):

    # Ensure that output corresponding to a known input processed
    # with a preset random seed remains consistent.
    optimizer = Optimizer(sample_random_vector, valid_srom)

    samples, probabilities = optimizer.get_optimal_params(num_test_samples=10,
                                                          verbose=False)

    assert np.allclose([np.sum(probabilities)], [1.])


def test_get_joint_optimal_params(sample_random_vector, valid_srom):

    # Ensure that output corresponding to a known input processed
    # with a preset random seed remains consistent.
    optimizer = Optimizer(sample_random_vector, valid_srom, joint_opt=True, scale=0.1)

    samples, probabilities = optimizer.get_optimal_params(num_test_samples=10,
                                                          joint_opt=True,
                                                          verbose=True)
    assert np.allclose([np.sum(probabilities)], [1.])
