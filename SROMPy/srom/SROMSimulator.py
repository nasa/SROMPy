import numpy as np

from SROMPy.target.RandomVariable import RandomVariable
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

        input_srom = self._generate_input_srom(srom_size, dim)

        if surrogate_type == "PWC":
            output_samples = \
                self._simulate_piecewise_constant(input_srom)
            
            output_gradients = None

        elif surrogate_type == "PWL":
            output_samples, output_gradients = \
                self._simulate_piecewise_linear(srom_size, 
                                                input_srom,
                                                pwl_step_size)

        srom_surrogate = \
            SROMSurrogate(input_srom, output_samples, output_gradients)

        return srom_surrogate
    
    def _simulate_piecewise_constant(self, input_srom):
        srom_output, _ = \
            self.evaluate_model_for_samples(input_srom)

        return srom_output
    
    def _simulate_piecewise_linear(self, srom_size, input_srom, pwl_step_size):        
        srom_output, samples = \
            self.evaluate_model_for_samples(input_srom)

        samples_fd = \
            FD.get_perturbed_samples(samples,
                                     perturbation_values=[pwl_step_size])

        gradient = \
            self._compute_pwl_gradient(srom_output,
                                       samples_fd,
                                       pwl_step_size,
                                       input_srom)

        return srom_output, gradient

    def _compute_pwl_gradient(self, srom_output, samples_fd, step_size,
                              input_srom):

        perturbed_output, _ = \
            self.evaluate_model_for_samples(input_srom, samples_fd)

        gradient = FD.compute_gradient(srom_output,
                                       perturbed_output,
                                       [step_size])

        return gradient
    
    def evaluate_model_for_samples(self, input_srom, samples_fd=None):
        if samples_fd is not None:
            samples = samples_fd
        else:
            (samples, _) = input_srom.get_params()

        output = np.zeros(len(samples))

        for i, values in enumerate(samples):
            output[i] = self._model.evaluate([values])

        return output, samples

    def _generate_input_srom(self, srom_size, dim):
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
