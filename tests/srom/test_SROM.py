import pytest
import numpy as np

from src.srom.SROM import SROM
from src.target import SampleRandomVector
from src.target import RandomVector

class TestSROM:

    def __init__(self):

        self.sample_random_vector = SampleRandomVector(np.zeros(10))

    def test_invalid_init_parameter_values_rejected(self):

        # Ensure exception for invalid srom dimensions.
        with pytest.raises(ValueError):
            SROM(0, 1)
            SROM(-1, 1)
            SROM(10, 0)
            SROM(10, 3)

    def test_invalid_optimize_parameter_values_rejected(self):

        pass