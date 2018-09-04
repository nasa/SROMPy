import pytest

from src.target import UniformRandomVariable

class TestUniformRandomVariable():

    def test_raises_exception_for_invalid_variable_range(self):

        minimum_value = 5
        maximum_value = 3

        with pytest.raises(ValueError):
            uniform = UniformRandomVariable(minimum_value, maximum_value)

