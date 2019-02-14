
from SROMPy.target import RandomVariable
from SROMPy.srom import Model

class SROMSimulator(object):

    def __init__(self, random_input, model):
        self.__check_init_parameters(random_input, model)

        self._data = random_input
        self._models = model

    def simulate(self, srom_size, surrogate_type):
        self.__check_simulate_parameters(srom_size, surrogate_type)
    
    @staticmethod
    def __check_init_parameters(data, model):
        if not isinstance(data, RandomVariable):
            TypeError("Data must inherit from the RandomEntity class")
        
        if not isinstance(model, Model):
            TypeError("Model must inherit from Model class")

    @staticmethod
    def __check_simulate_parameters(size, surrogate_type):
        if not isinstance(size, int):
            TypeError("SROM size must be an integer")
        
        #Update surrogate type exception (TODO)
        #if surrogate_type == "PWC":
        #    TypeError("For now, surrogate type must PWC")
