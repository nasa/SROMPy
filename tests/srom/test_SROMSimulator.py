import pytest
import os
import sys
import numpy as np

if 'PYTHONPATH' not in os.environ:
    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from SROMPy.srom.SROMSimulator import SROMSimulator
from SROMPy.target import BetaRandomVariable
from spring_mass_model import SpringMassModel

@pytest.fixture
def beta_random_variable_distribution():
    np.random.seed(1)

    random_input = \
        BetaRandomVariable(alpha=3.0, beta=2.0, shift=1.0, scale=2.5)

    return random_input

@pytest.fixture
def spring_model():
    model = SpringMassModel(mass=1.5, time_step=1.0, state0=[0.0, 0.0])

    return model

@pytest.fixture
def srom_simulator_instance(beta_random_variable_distribution, spring_model):
    srom_sim = SROMSimulator(beta_random_variable_distribution, spring_model)

    return srom_sim

#Come up with a better test name (TODO)
def test_simulator_init_exceptions(beta_random_variable_distribution, 
                                   spring_model):

    with pytest.raises(TypeError):
        SROMSimulator(random_input=5.5, model=spring_model)
    
    with pytest.raises(TypeError):
        SROMSimulator(beta_random_variable_distribution, "TestString")

def test_simulate_function_exceptions(srom_simulator_instance):
    with pytest.raises(TypeError):
        srom_simulator_instance.simulate(srom_size=10.5, surrogate_type="PWC")

    with pytest.raises(ValueError):
        srom_simulator_instance.simulate(srom_size=10, surrogate_type="JBC")
