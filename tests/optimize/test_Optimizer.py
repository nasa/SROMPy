import pytest
import numpy as np

from src.optimize.Optimizer import Optimizer
from src.srom.SROM import SROM
from src.target.SampleRandomVector import SampleRandomVector


class TestOptimizer():

    def __init__(self):

        np.random.seed(1)
        random_vector = np.random.rand(10)
        self.sample_random_vector = SampleRandomVector(random_vector)
        self.valid_srom = SROM(10, 1)

    def test_invalid_init_parameter_values_rejected(self):

        # Ensure no exception using default parameters.
        Optimizer(self.sample_random_vector, self.valid_srom)

        # Ensure exception for invalid target parameter.
        with pytest.raises(TypeError):
            Optimizer([], self.valid_srom)
            Optimizer(np.zeros(10), self.valid_srom)

        with pytest.raises(ValueError):
            Optimizer(None, self.valid_srom)

    def test_invalid_get_optimal_params_parameter_values_rejected(self):

        pass

    def test_get_optimal_params_expected_output(self):

        optimizer = Optimizer(self.sample_random_vector,
                              self.valid_srom)

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
