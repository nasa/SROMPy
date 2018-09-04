import numpy as np
import pytest

from src.optimize.Gradient import Gradient
from src.target.SampleRandomVector import SampleRandomVector


class TestGradient():

    def test_invalid_init_parameter_values_rejected(self):

        sample_random_vector = SampleRandomVector(np.zeros(10))

        # Ensure no exception using default parameters.
        Gradient(SROM=None, targetRV=sample_random_vector)

        # Ensure exception if obj_weights parameter is too small.
        with pytest.raises(ValueError):
            Gradient(SROM=None, targetRV=sample_random_vector, obj_weights=np.ones((2,)))

        # Ensure exception if specified error parameter is invalid.
        with pytest.raises(ValueError):
            Gradient(SROM=None, targetRV=sample_random_vector, error='test')

