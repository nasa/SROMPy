import abc

'''
Abstract class defining the target random vector being matched by an SROM.
Inherited by AnalyticRandomVector and SampleRandomVector to define analytically specified and 
sample-based random vectors, respectively.
'''


class RandomVector(object):

    def __init__(self, dim):

        self._dim = int(dim)

    @abc.abstractmethod
    def compute_moments(self, max_order):
        return
    
    @abc.abstractmethod
    def compute_CDF(self, x_grid):
        return

    @abc.abstractmethod
    def compute_corr_mat(self):
        return

    @abc.abstractmethod
    def draw_random_sample(self, sample_size):
        return

    @abc.abstractmethod
    def get_dim(self):
        return self._dim
