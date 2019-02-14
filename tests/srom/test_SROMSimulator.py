import pytest
import os
import sys
import numpy as np

if 'PYTHONPATH' not in os.environ:
    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from SROMPy.srom.SROMSimulator import SROMSimulator
from SROMPy.target import BetaRandomVariable
from SROMPy.srom.spring_mass_model import SpringMassModel

@pytest.fixture
def beta_random_variable():
    random_input = \
        BetaRandomVariable(alpha=3.0, beta=2.0, shift=1.0, scale=2.5)

    return random_input

@pytest.fixture
def spring_model_fixture():
    spring_model = SpringMassModel(state0=[0.0, 0.0], time_step=0.01)

    return spring_model

@pytest.fixture
def srom_simulator_fixture():
    random_variable = \
        BetaRandomVariable(alpha=3.0, beta=2.0, shift=1.0, scale= 2.5)

    spring_model = SpringMassModel(state0=[0.0, 0.0], time_step=0.01)

    srom_sim = SROMSimulator(random_variable, spring_model)
    return srom_sim

def test_simulator_init_exception_for_invalid_parameters(beta_random_variable, 
                                                         spring_model_fixture):

    with pytest.raises(TypeError):
        SROMSimulator(1, spring_model_fixture)

    with pytest.raises(TypeError):
        SROMSimulator(beta_random_variable, "Not A Proper Model")

def test_simulate_exception_for_invalid_parameters(srom_simulator_fixture):
    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10.5, 1,"PWC")

    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10, 1.5,"PWC")

    with pytest.raises(ValueError):
        srom_simulator_fixture.simulate(10, 1,"Not A Proper Model")

