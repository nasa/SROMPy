import numpy as np

from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.srom.Model import Model
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM

class SROMSimulator:

    def __init__(self, random_input, model):
        self.__check_init_parameters(random_input, model)

        self._data = random_input
        self._model = model

    #Checks to see what surrogate type, then calls correct fxn
    def simulate(self, srom_size, dim, surrogate_type):
        self.__check_simulate_parameters(srom_size, surrogate_type)

        if surrogate_type == "PWC":
            self._simulate_piecewise_computation(srom_size, dim)

    
    def _simulate_piecewise_computation(self, srom_size, dim):
        input_srom = self.__instantiate_srom(srom_size, dim)
        self._postprocessor_input(input_srom)

        srom_displacements, probabilities = \
            self._get_srom_max_displacement(srom_size, input_srom)
        
        output_srom = self.__instantiate_srom(srom_size, dim)
        output_srom.set_params(srom_displacements, probabilities)

    #Check to make sure it is returning correct data (TODO)
    def _postprocessor_input(self, input_srom):
        pp_input = \
            Postprocessor(srom=input_srom,target_random_vector=self._data)

        print "Computing piecewise constant SROM approximation for output..."

        return pp_input.compare_cdfs()

    #Check what values it is returning and possibly the get_params (TODO)
    def _get_srom_max_displacement(self, srom_size, input_srom):
        srom_displacements = np.zeros(srom_size)

        (samples, probabilities) = input_srom.get_params()

        for i, values in enumerate(samples):
            srom_displacements[i] = self._model.evaluate([values])
        
        return srom_displacements, probabilities


    @staticmethod
    def __check_init_parameters(data, model):
        if not isinstance(data, RandomVariable):
            TypeError("Data must inherit from the RandomVariable class")
        
        if not isinstance(model, Model):
            TypeError("Model must inherit from Model class")

    #Test to make sure it is throwing exceptions, fix problem with except (TODO)
    @staticmethod
    def __check_simulate_parameters(size, surrogate_type):
        if not isinstance(size, int):
            TypeError("SROM size must be an integer")
        
        #Update surrogate type exception (TODO)
        if surrogate_type != "PWC":
            ValueError("For now, surrogate type must PWC")

    #Test to make sure it returns SROM (TODO)
    @staticmethod
    def __instantiate_srom(srom_size, dim):
        srom = SROM(srom_size, dim)

        return srom
