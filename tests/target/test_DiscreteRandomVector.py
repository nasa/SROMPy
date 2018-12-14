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


from SROMPy.target.DiscreteRandomVector import DiscreteRandomVector



@pytest.fixture
def simple_discrete_rv_1d():

    samples = np.array([1,2,3,4])
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])

    discrete_rv = DiscreteRandomVector(samples, probabilities)
    return discrete_rv

@pytest.fixture
def simple_discrete_rv_2d():

    samples = np.array([[1,4],
                        [2,5],
                        [3,6],
                        [4,7]])
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])

    discrete_rv = DiscreteRandomVector(samples, probabilities)
    return discrete_rv


def test_compute_moments_simple_1d(simple_discrete_rv_1d):

    moments = simple_discrete_rv_1d.compute_moments(max_order=2)

    #Check first two moments
    assert np.isclose(moments[0], 2.5)
    assert np.isclose( moments[1], 7.5)
    assert len(moments) == 2

def test_compute_cdfs_simple_1d(simple_discrete_rv_1d):

    x_grid = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    cdf_vals = simple_discrete_rv_1d.compute_cdf(x_grid)

    true_cdf_vals = [0., 0.25, 0.5, 0.75, 1.0]

    for i, true_val in enumerate(true_cdf_vals):
        assert np.isclose(cdf_vals[i], true_val)


def test_compute_moments_simple_2d(simple_discrete_rv_2d):

    moments = simple_discrete_rv_2d.compute_moments(max_order=2)

    #Check mean and variance in both dimensions:
    assert np.isclose(moments[0,0], 2.5)
    assert np.isclose( moments[1,0], 7.5)
    assert np.isclose(moments[0,1], 5.5)
    assert np.isclose( moments[1,1], 31.5)

def test_compute_cdfs_simple_2d(simple_discrete_rv_2d):

    #CDFs are evaluated on the same x-grid for each dimension
    x_grid = np.array([0.5, 2.5, 4.5, 6.5, 7.5])
    cdf_vals = simple_discrete_rv_2d.compute_cdf(x_grid)

    true_cdf_vals_dim1 = [0., 0.5, 1.0, 1.0, 1.0]
    for i, true_val in enumerate(true_cdf_vals_dim1):
        assert np.isclose(cdf_vals[i,0], true_val)

    true_cdf_vals_dim2 = [0., 0., 0.25, 0.75, 1.0]
    for i, true_val in enumerate(true_cdf_vals_dim2):
        assert np.isclose(cdf_vals[i,1], true_val)

def test_raise_value_error_for_bad_probabilities():

    samples = np.array([1,2,3,4])

    probs_wrong_dim = np.array([0.25, 0.25, 0.5])
    with pytest.raises(ValueError):
        discrete_rv = DiscreteRandomVector(samples, probs_wrong_dim)

    probs_wrong_sum = np.array([0.25, 0.25, 0.25, 0.1])
    with pytest.raises(ValueError):
        discrete_rv = DiscreteRandomVector(samples, probs_wrong_sum)

    probs_neg_value = np.array([0.5, 0.5, 0.25, -0.25 ])
    with pytest.raises(ValueError):
        discrete_rv = DiscreteRandomVector(samples, probs_neg_value)

def test_raise_value_error_for_bad_samples_1d():

    probabilities = np.array([0.25,0.25,0.25,0.25])

    samples_wrong_size = np.array([1,2,3])
    with pytest.raises(ValueError):
        discrete_rv = DiscreteRandomVector(samples_wrong_size, probabilities)

    samples_wrong_shape = np.array([[1,2],[3,4]])
    with pytest.raises(ValueError):
        discrete_rv = DiscreteRandomVector(samples_wrong_shape, probabilities)
        


