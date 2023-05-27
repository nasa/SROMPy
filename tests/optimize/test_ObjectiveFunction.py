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
from SROMPy.optimize import ObjectiveFunction
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
def valid_srom_smooth():
    srom = SROM(10, 1)
    srom._scale = 0.1
    return srom


def test_invalid_init_parameter_values_rejected(valid_srom,
                                                sample_random_vector):
    with pytest.raises(TypeError):
        ObjectiveFunction(srom="srom",
                          target=sample_random_vector,
                          obj_weights=None,
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(TypeError):
        ObjectiveFunction(srom=valid_srom,
                          target="victor",
                          obj_weights=None,
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(TypeError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights="heavy",
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(TypeError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=None,
                          error=1.,
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(TypeError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=None,
                          error="MEAN",
                          max_moment="first",
                          num_cdf_grid_points=100)

    with pytest.raises(TypeError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=None,
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=[1, 2])

    with pytest.raises(ValueError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=np.zeros((5, 2)),
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(ValueError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=np.zeros(5),
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    with pytest.raises(ValueError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=np.ones(3) * -1,
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)

    sample_random_vector._dim = 0
    with pytest.raises(ValueError):
        ObjectiveFunction(srom=valid_srom,
                          target=sample_random_vector,
                          obj_weights=None,
                          error="MEAN",
                          max_moment=2,
                          num_cdf_grid_points=100)


def test_evaluate_returns_expected_result(valid_srom, sample_random_vector):
    samples = np.ones((valid_srom._size, valid_srom._dim))
    probabilities = np.ones(valid_srom._size)

    for objective_weights in [[0., .05, 1.], [7., .4, .1]]:
        for error_function in ["mean", "max", "sse"]:
            for max_moment in [1, 2, 3, 4]:
                for num_cdf_grid_points in [2, 15, 70]:
                    objective_function = \
                        ObjectiveFunction(srom=valid_srom,
                                          target=sample_random_vector,
                                          obj_weights=objective_weights,
                                          error=error_function,
                                          max_moment=max_moment,
                                          num_cdf_grid_points=
                                          num_cdf_grid_points)

                    error = objective_function.evaluate(samples, probabilities)

                    assert isinstance(error, float)
                    assert error > 0.


def test_evaluate_returns_expected_result_smooth(valid_srom_smooth, sample_random_vector):
    samples = np.ones((valid_srom_smooth._size, valid_srom_smooth._dim))
    probabilities = np.ones(valid_srom_smooth._size)

    for objective_weights in [[0., .05, 1.], [7., .4, .1]]:
        for error_function in ["mean", "max", "sse"]:
            for max_moment in [1, 2, 3, 4]:
                for num_cdf_grid_points in [2, 15, 70]:
                    objective_function = \
                        ObjectiveFunction(srom=valid_srom_smooth,
                                          target=sample_random_vector,
                                          obj_weights=objective_weights,
                                          error=error_function,
                                          max_moment=max_moment,
                                          num_cdf_grid_points=
                                          num_cdf_grid_points)

                    error = objective_function.evaluate(samples, probabilities)

                    assert isinstance(error, float)
                    assert error > 0.
