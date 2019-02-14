
from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.srom.Model import Model
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM

class SROMSimulator:

    def __init__(self, random_input, model):
        self.__check_init_parameters(random_input, model)

        self._data = random_input
        self._models = model

    def simulate(self, srom_size, dim, surrogate_type):
        self.__check_simulate_parameters(srom_size, surrogate_type)

        if surrogate_type == "PWC":
            input_srom = self._instantiate_srom(srom_size, dim)
            self._postprocessor_input(input_srom)

    def _postprocessor_input(self, input_srom):
        pp_input = Postprocessor(srom=input_srom,target_random_vector=self._data)

        return pp_input.compare_cdfs()

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
    def _instantiate_srom(srom_size, dim):
        srom = SROM(srom_size, dim)

        return srom
