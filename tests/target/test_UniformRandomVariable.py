import pytest

from src.target import UniformRandomVariable


def test_raises_exception_for_invalid_variable_range():

    minimum_value = 5
    maximum_value = 3

    with pytest.raises(ValueError):
        UniformRandomVariable(minimum_value, maximum_value)

