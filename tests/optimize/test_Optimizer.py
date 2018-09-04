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


def test_invalid_init_parameter_values_rejected(sample_random_vector, valid_srom):

    # Ensure no exception using default parameters.
    Optimizer(sample_random_vector, valid_srom)

    # Ensure exception for invalid target parameter.
    with pytest.raises(TypeError):
        Optimizer([], valid_srom)
        Optimizer(np.zeros(10), valid_srom)

    with pytest.raises(ValueError):
        Optimizer(None, valid_srom)


def test_invalid_get_optimal_params_parameter_values_rejected():

    pass


def test_get_optimal_params_expected_output(sample_random_vector, valid_srom):

    optimizer = Optimizer(sample_random_vector, valid_srom)

    samples, probs = optimizer.get_optimal_params(num_test_samples=10)
    print samples
    print probs

    expected_samples = np.array([[3.96767474e-01],
                                 [3.02332573e-01],
                                 [9.23385948e-02],
                                 [5.38816734e-01],
                                 [4.17022005e-01],
                                 [1.86260211e-01],
                                 [7.20324493e-01],
                                 [3.45560727e-01],
                                 [1.46755891e-01],
                                 [1.14374817e-04]])

    assert np.sum(probs) == 1.
    assert np.allclose(samples, expected_samples)
