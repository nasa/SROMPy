import numpy as np
import pytest

from src.optimize.Gradient import Gradient
from src.target.SampleRandomVector import SampleRandomVector


class TestGradient():

    def test_init_exception_if_obj_weights_param_too_small(self):

        sample_random_vector = SampleRandomVector(np.zeros(10))
        with pytest.raises(ValueError):
            Gradient(SROM=None, targetRV=sample_random_vector, obj_weights=np.ones((2,)))

    def test_init_exception_if_error_param_invalid(self):

        sample_random_vector = SampleRandomVector(np.zeros(10))
        with pytest.raises(ValueError):
            Gradient(SROM=None, targetRV=sample_random_vector, error='test')

    def test_failure(self):

        assert 1 == 2



