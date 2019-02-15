import numpy as np

from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.target import SampleRandomVector
from SROMPy.srom.Model import Model
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate

class SROMSimulator:

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

        self._postprocessor_input(input_srom)

        srom_displacements, probabilities = \
            self._get_srom_max_displacement(srom_size, input_srom)[:2]
        
        #The way this wraps looks ugly, but is extremely low priority (TODO)
        self._output_srom_results(srom_size, dim, srom_displacements, 
                                  probabilities)

    def _simulate_piecewise_linear(self, srom_size, dim, pwl_step_size):
        input_srom = self.__instantiate_srom(srom_size, dim)

        srom_displacements, samples = \
            self._get_pwl_samples(srom_size, input_srom)

        samples_fd = \
            FD.get_perturbed_samples(samples=samples, 
                                     perturbation_values=[pwl_step_size])

        self._compute_pwl_gradient(srom_displacements, srom_size, 
                                   samples_fd, pwl_step_size)

    #Check to make sure it is returning correct data (TODO)
    def _postprocessor_input(self, input_srom):
        pp_input = \
            Postprocessor(srom=input_srom, target_random_vector=self._data)

        pp_input.compare_cdfs()

    #Check what values it is returning and possibly the get_params (TODO)
    def _get_srom_max_displacement(self, srom_size, input_srom):
        srom_displacements = np.zeros(srom_size)

        (samples, probabilities) = input_srom.get_params()

        for i, values in enumerate(samples):
            srom_displacements[i] = self._model.evaluate([values])

        return srom_displacements, probabilities, samples

    def _get_pwl_samples(self, srom_size, input_srom):
        srom_displacements, samples = \
            self._get_srom_max_displacement(srom_size, input_srom)[::2]
        
        return srom_displacements, samples

    #MAKE A VALUES ENUMERATE FUNCTION (TODO)
    def _compute_pwl_gradient(self, srom_displacements, 
                              srom_size, samples_fd, step_size):

        perturbed_displacements = np.zeros(srom_size)

        for i, values in enumerate(samples_fd):
            perturbed_displacements[i] = self._model.evaluate([values])

        gradient = FD.compute_gradient(srom_displacements, 
                                       perturbed_displacements, 
                                       [step_size])

        return gradient

    def _output_srom_results(self, srom_size, dim, 
                             displacement_samples, probabilities):

        output_srom = self.__instantiate_srom(srom_size, dim)
        output_srom.set_params(displacement_samples, probabilities)

        self._postprocessor_output(output_srom)
        self.__print_mean_comparison(displacement_samples, output_srom)

    def _postprocessor_output(self, output_srom):
        monte_carlo_solution = self._generate_monte_carlo_solution()

        pp_output = Postprocessor(output_srom, monte_carlo_solution)
        pp_output.compare_cdfs()

    def _generate_monte_carlo_solution(self):
        num_samples = 5000
        monte_carlo_samples = self._data.draw_random_sample(num_samples)

        displacement_samples = np.zeros(num_samples)
        for i, values in enumerate(monte_carlo_samples):
            displacement_samples[i] = self._model.evaluate([values])
        
        monte_carlo_solution = SampleRandomVector(displacement_samples)
        
        return monte_carlo_solution

    @staticmethod
    def __check_init_parameters(data, model):
        if not isinstance(data, RandomVariable):
            raise TypeError("Data must inherit from the RandomVariable class")
        
        if not isinstance(model, Model):
            raise TypeError("Model must inherit from Model class")

    #Test to make sure it is throwing exceptions, fix problem with except (TODO)
    @staticmethod
    def __check_simulate_parameters(srom_size, dim, surrogate_type, pwl_step_size):
        if not isinstance(srom_size, int):
            raise TypeError("SROM size must be an integer.")

        #Check if dim is short for dimensions (TODO)
        if not isinstance(dim, int):
            raise TypeError("Dimensions must be an integer.")

        #Update surrogate type exception (TODO)
        if surrogate_type != "PWC" and surrogate_type != "PWL":
            raise ValueError("For now, surrogate type must PWC.")
        
        #Come up with a better error code (TODO)
        if surrogate_type == "PWL" and isinstance(pwl_step_size, None):
            raise ValueError("Step size must be instantiated for PWL")
    #Test to make sure it returns SROM (TODO)
    @staticmethod
    def __instantiate_srom(srom_size, dim):
        srom = SROM(srom_size, dim)

        return srom

    @staticmethod
    def __print_mean_comparison(displacement_samples, output_srom):
        print
        print "Monte Carlo mean estimate: ", np.mean(displacement_samples)
        print "SROM mean estimate: ", output_srom.compute_moments(1)[0][0]
