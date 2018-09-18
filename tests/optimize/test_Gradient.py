import numpy as np
import pytest

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


def test_invalid_init_parameter_values_rejected(sample_random_vector, valid_srom):

    # Ensure no exception using default parameters.
    Gradient(SROM=valid_srom, targetRV=sample_random_vector)

    # Ensure exception if obj_weights parameter is too small.
    with pytest.raises(ValueError):
        Gradient(SROM=valid_srom, targetRV=sample_random_vector, obj_weights=np.ones((2,)))

    # Ensure exception if specified error parameter is invalid.
    with pytest.raises(ValueError):
        Gradient(SROM=valid_srom, targetRV=sample_random_vector, error='test')

