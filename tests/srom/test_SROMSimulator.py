import pytest
import os
import sys
import numpy as np

if 'PYTHONPATH' not in os.environ:
    base_path = os.path.abspath('..')

    sys.path.insert(0, base_path)

from SROMPy.target import BetaRandomVariable, SampleRandomVector
from SROMPy.srom import SROM, SROMSurrogate, FiniteDifference as FD
from SROMPy.srom.spring_mass_model import SpringMassModel
from SROMPy.srom.SROMSimulator import SROMSimulator

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

@pytest.fixture
def srom_base_fixture():
    srom = SROM(size=10, dim=1)
    
    return srom

def test_simulator_init_exception_for_invalid_parameters(beta_random_variable, 
                                                         spring_model_fixture):

    with pytest.raises(TypeError):
        SROMSimulator(1, spring_model_fixture)

    with pytest.raises(TypeError):
        SROMSimulator(beta_random_variable, "Not A Proper Model")

def test_simulate_exception_for_invalid_parameters(srom_simulator_fixture):
    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10.5, 1,"PWC", 1e-12)

    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10, 1.5,"PWL", 1e-12)

    with pytest.raises(ValueError):
        srom_simulator_fixture.simulate(10, 1, "no", 1e-12)

    with pytest.raises(TypeError):
        srom_simulator_fixture.simulate(10, 1, "PWL")


def test_simulate_pwc_spring_mass(srom_simulator_fixture):
    '''
    Tests a PWC surrogate against a manual reference solution generated from  
    test_scripts_data/generate_srom_sim_ref_solution.py
    '''

    pwc_surrogate = srom_simulator_fixture.simulate(10, 1, "PWC")

    mean_pwc = pwc_surrogate.compute_moments(1)[0][0]
    mean_reference = 12.385393457327542

    assert np.isclose(mean_pwc, mean_reference)

def test_simulate_pwl_spring_mass(srom_simulator_fixture):
    '''
    Tests a PWL surrogate against a manual reference solution generated from  
    test_scripts_data/generate_srom_sim_ref_solution.py
    '''

    pwl_surrogate = srom_simulator_fixture.simulate(10, 1, "PWL", 1e-12)

    output_pwl = pwl_surrogate.sample(np.array([2]))
    output_ref = np.array([[14.69958116]])

    assert np.isclose(output_pwl, output_ref)


def test_srom_displacement_return_type(srom_simulator_fixture, 
                                       beta_random_variable):
    srom_size = 10
    dim = 1
    input_srom = SROM(srom_size, dim)
    input_srom.optimize(beta_random_variable)

    displacements, samples = \
        srom_simulator_fixture._srom_max_displacement(srom_size, input_srom)

    assert isinstance(displacements, np.ndarray)
    assert isinstance(samples, np.ndarray)

def test_compute_pwl_gradient_return_type(srom_simulator_fixture,
                                          beta_random_variable,
                                          srom_base_fixture):
    srom_size = 10
    pwl_step_size = 1e-12
    srom_base_fixture.optimize(beta_random_variable)

    displacements, samples = \
        srom_simulator_fixture._srom_max_displacement(srom_size,
                                                      srom_base_fixture)
    samples_fd = \
        FD.get_perturbed_samples(samples,
                                 perturbation_values=[pwl_step_size])

    test_gradient = \
        srom_simulator_fixture._compute_pwl_gradient(displacements,
                                                     srom_size,
                                                     samples_fd,
                                                     pwl_step_size)
    assert isinstance(test_gradient, np.ndarray)

def test_simulate_return_type(srom_simulator_fixture):
    test_pwc_surrogate = \
        srom_simulator_fixture.simulate(srom_size=10,
                                        dim=1,
                                        surrogate_type="PWC")

    test_pwl_surrogate = \
        srom_simulator_fixture.simulate(srom_size=10,
                                        dim=1,
                                        surrogate_type="PWL",
                                        pwl_step_size=1e-12)

    assert isinstance(test_pwc_surrogate, SROMSurrogate)
    assert isinstance(test_pwl_surrogate, SROMSurrogate)
