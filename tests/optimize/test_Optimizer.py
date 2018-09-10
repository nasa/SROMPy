import pytest
import numpy as np

from src.optimize import Optimizer
from src.srom import SROM
from src.target import SampleRandomVector


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
    with pytest.raises(TypeError):
        Optimizer(sample_random_vector, valid_srom, [1., 1., 1.])

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


def test_invalid_get_optimal_params_parameter_values_rejected(sample_random_vector,
                                                              valid_srom):

    # Ensure no errors with valid parameters.
    optimizer = Optimizer(sample_random_vector, valid_srom)

    # Ensure num_test_samples is positive integer.
    with pytest.raises(TypeError):
        optimizer.get_optimal_params(num_test_samples="One")

    with pytest.raises(ValueError):
        optimizer.get_optimal_params(num_test_samples=0)


def test_get_optimal_params_expected_output(sample_random_vector, valid_srom):

    # Ensure that output corresponding to a known input processed
    # with a preset random seed remains consistent.
    optimizer = Optimizer(sample_random_vector, valid_srom)

    samples, probs = optimizer.get_optimal_params(num_test_samples=10,
                                                  verbose=False)

    assert np.allclose([np.sum(probs)], [1.])
