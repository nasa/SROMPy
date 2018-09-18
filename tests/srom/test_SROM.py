import pytest
import numpy as np

from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector
from SROMPy.target import RandomVector


def test_invalid_init_parameter_values_rejected():

    # Ensure exception for invalid srom dimensions.
    with pytest.raises(ValueError):
        SROM(0, 1)
        SROM(-1, 1)
        SROM(10, 0)
        SROM(10, 3)


def test_invalid_optimize_parameter_values_rejected():

    pass
