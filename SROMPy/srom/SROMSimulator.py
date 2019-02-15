import numpy as np

from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.target import SampleRandomVector
from SROMPy.srom.Model import Model
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate

class SROMSimulator(object):

    def __init__(self, random_input, model):
        self.__check_init_parameters(random_input, model)

        self._data = random_input
        self._model = model

    #Checks to see what surrogate type, then calls correct fxn
    def simulate(self, srom_size, dim, surrogate_type, pwl_step_size=None):
        #This wrapping looks ugly but low priority (TODO)
        self.__check_simulate_parameters(srom_size, dim,
                                         surrogate_type, pwl_step_size)

        if surrogate_type == "PWC":
            self._simulate_piecewise_computation(srom_size, dim)
        elif surrogate_type == "PWL":
            self._simulate_piecewise_linear(srom_size, dim, pwl_step_size)

    def _simulate_piecewise_computation(self, srom_size, dim):
        input_srom = SROM(srom_size, dim)
        input_srom.optimize(self._data)

        srom_displacements, probabilities, _ = \
            self._srom_max_displacement(srom_size, input_srom)

        self._postprocessor_input(input_srom)

        #The way this wraps looks ugly, but is extremely low priority (TODO)
        self._output_srom_results(srom_size, dim, srom_displacements,
                                  probabilities)

    def _simulate_piecewise_linear(self, srom_size, dim, pwl_step_size):
        input_srom = self.__instantiate_srom(srom_size, dim)
        input_srom.optimize(self._data)

        srom_displacements, samples = \
             self._pwl_samples(srom_size, input_srom)

        samples_fd = \
            FD.get_perturbed_samples(samples=samples,
                                     perturbation_values=[pwl_step_size])

        gradient = \
            self._compute_pwl_gradient(srom_displacements,
                                       srom_size,
                                       samples_fd,
                                       pwl_step_size)

        pwl_surrogate = \
            self.__instantiate_srom_surrogate(input_srom,
                                              srom_displacements,
                                              gradient)

        output_samples = \
            self._pwl_output_samples(pwl_surrogate)

        pwl_solution = \
            self.__pwl_solution(output_samples)

        self._postprocessor_output(pwl_solution)

    #Check to make sure it is returning correct data (TODO)
    def _postprocessor_input(self, input_srom):
        pp_input = \
            Postprocessor(srom=input_srom, target_random_vector=self._data)

        pp_input.compare_cdfs()

    def _srom_max_displacement(self, srom_size, input_srom):
        (samples, probabilities) = input_srom.get_params()

        srom_displacements = \
            self._enumerate_utility_function(srom_size, samples)

        return srom_displacements, probabilities, samples

    def _enumerate_utility_function(self, srom_size, samples):
        displacements = np.zeros(srom_size)

        for i, values in enumerate(samples):
            displacements[i] = self._model.evaluate([values])

        return displacements

    def _pwl_samples(self, srom_size, input_srom):
        srom_displacements, _, samples = \
            self._srom_max_displacement(srom_size, input_srom)

        return srom_displacements, samples

    def _compute_pwl_gradient(self, srom_displacements,
                              srom_size, samples_fd, step_size):

        perturbed_displacements = \
            self._enumerate_utility_function(srom_size, samples_fd)

        gradient = FD.compute_gradient(srom_displacements,
                                       perturbed_displacements,
                                       [step_size])

        return gradient

    def _pwl_output_samples(self, pwl_surrogate):
        num_samples = 5000

        data_samples = self._data.draw_random_sample(num_samples)
        output_samples = \
            pwl_surrogate.sample(data_samples)

        return output_samples

    def _output_srom_results(self, srom_size, dim,
                             displacement_samples, probabilities):

        output_srom = self.__instantiate_srom(srom_size, dim)
        output_srom.set_params(displacement_samples, probabilities)

        self._postprocessor_output(output_srom)

    def _postprocessor_output(self, output_srom):
        monte_carlo_solution, displacement_samples = \
            self._generate_monte_carlo_solution()

        self.__print_mean_comparison(displacement_samples, output_srom)

        pp_output = Postprocessor(output_srom, monte_carlo_solution)
        pp_output.compare_cdfs()

    def _generate_monte_carlo_solution(self):
        num_samples = 5000
        monte_carlo_samples = self._data.draw_random_sample(num_samples)

        displacement_samples = \
            self._enumerate_utility_function(num_samples, monte_carlo_samples)

        monte_carlo_solution = SampleRandomVector(displacement_samples)

        return monte_carlo_solution, displacement_samples

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

        #Check if dim is short for dimensions (TODO)
        if not isinstance(dim, int):
            raise TypeError("Dimensions must be an integer")

        if surrogate_type != "PWC" and surrogate_type != "PWL":
            raise ValueError("Surrogate type must be 'PWC' or 'PWL'")

        if surrogate_type == "PWL" and pwl_step_size is None:
            raise TypeError("Step size must be initialized for 'PWL' ex: 1e-12")

    @staticmethod
    def __instantiate_srom(srom_size, dim):
        srom = SROM(srom_size, dim)

        return srom

    @staticmethod
    def __instantiate_srom_surrogate(input_srom, srom_displacements, gradient):
        pwl_surrogate = SROMSurrogate(input_srom, srom_displacements, gradient)

        return pwl_surrogate

    @staticmethod
    def __pwl_solution(output_samples):
        pwl_solution = SampleRandomVector(output_samples)

        return pwl_solution

    @staticmethod
    def __print_mean_comparison(displacement_samples, output_srom):
        print
        print "Monte Carlo mean estimate: ", np.mean(displacement_samples)
        print "SROM mean estimate: ", output_srom.compute_moments(1)[0][0]
