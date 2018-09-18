import abc

'''
Abstract class defining the target random variable being matched by an SROM.
Inherited by BetaRandomVariable and GammaRandomVariable, and NormalRandomVariable.
'''


class RandomVariable(object):

    @abc.abstractmethod
    def get_variance(self, max_order):
        return
    
    @abc.abstractmethod
    def compute_moments(self, x_grid):
        return

    @abc.abstractmethod
    def compute_CDF(self):
        return

    @abc.abstractmethod
    def compute_inv_CDF(self, sample_size):
        return

    @abc.abstractmethod
    def compute_pdf(self):
        return

    @abc.abstractmethod
    def draw_random_sample(self):
        return

    @abc.abstractmethod
    def generate_moments(self):
        return

    @abc.abstractmethod
    def get_dim(self):
        return
