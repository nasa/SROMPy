import numpy as np

from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.target import SampleRandomVector
from SROMPy.srom.Model import Model
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate

class SROMSimulator(object):

    def __init__(self, random_input, model):
        self.__check_init_parameters(random_input, model)

        self._random_variable_data = random_input
        self._model = model
    
    def simulate(self, srom_size, dim, surrogate_type, pwl_step_size=None):
        self.__check_simulate_parameters(srom_size, 
                                        dim,
                                        surrogate_type,
                                        pwl_step_size)

        if surrogate_type == "PWC":
            input_srom, output_samples = \
                self._simulate_piecewise_computation(srom_size, dim)
            
            output_gradients = None

        elif surrogate_type == "PWL":
            input_srom, output_samples, output_gradients = \
                self._simulate_piecewise_linear(srom_size, 
                                                dim,
                                                pwl_step_size)

        srom_surrogate = \
            SROMSurrogate(input_srom, output_samples, output_gradients)

        return srom_surrogate
    
    def _simulate_piecewise_computation(self, srom_size, dim):
        input_srom = self._instantiate_srom(srom_size, dim)
        
        srom_displacements, _ = \
            self._srom_max_displacement(srom_size, input_srom)

        return input_srom, srom_displacements
    
    def _simulate_piecewise_linear(self, srom_size, dim, pwl_step_size):
        input_srom = self._instantiate_srom(srom_size, dim)
        
        srom_displacements, samples = \
            self._srom_max_displacement(srom_size, input_srom)

        samples_fd = \
            FD.get_perturbed_samples(samples,
                                     perturbation_values=[pwl_step_size])

        gradient = \
            self._compute_pwl_gradient(srom_displacements,
                                       srom_size,
                                       samples_fd,
                                       pwl_step_size)

        return input_srom, srom_displacements, gradient

    def _compute_pwl_gradient(self, srom_displacements, srom_size,
                              samples_fd, step_size):

        perturbed_displacements, _ = \
            self._enumerate_utility_function(srom_size, samples_fd)

        gradient = FD.compute_gradient(srom_displacements,
                                       perturbed_displacements,
                                       [step_size])

        return gradient

    def _srom_max_displacement(self, srom_size, input_srom):
        (samples, _) = input_srom.get_params()
        displacements = np.zeros(srom_size)
        
        displacements, samples = \
            self._enumerate_utility_function(srom_size, samples)

        return displacements, samples
    
    def _enumerate_utility_function(self, srom_size, samples):
        displacements = np.zeros(srom_size)

        for i, values in enumerate(samples):
            displacements[i] = self._model.evaluate([values])

        return displacements, samples

    def _instantiate_srom(self, srom_size, dim):
        srom = SROM(srom_size, dim)
        srom.optimize(self._random_variable_data)
        
        return srom

    @staticmethod
    def __check_init_parameters(data, model):
        if not isinstance(data, RandomVariable):
            raise TypeError("Data must inherit from the RandomVariable class")

        if not isinstance(model, Model):
            raise TypeError("Model must inherit from Model class")

    @staticmethod
    def __check_simulate_parameters(srom_size, dim, surrogate_type, 
                                    pwl_step_size):

        if not isinstance(srom_size, int):
            raise TypeError("SROM size must be an integer")

        if not isinstance(dim, int):
            raise TypeError("Dim must be an integer")

        if surrogate_type != "PWC" and surrogate_type != "PWL":
            raise ValueError("Surrogate type must be 'PWC' or 'PWL'")
        #Should this be a TypeError or ValueError? Leaning on value(TODO)
        if surrogate_type == "PWL" and pwl_step_size is None:
            raise TypeError("Step size must be initialized for 'PWL' ex: 1e-12")
