'''
Class for defining a uniform random variable
'''

import numpy as np
from scipy.stats import uniform as scipyuniform

from SROMPy.target.RandomVariable import RandomVariable


class UniformRandomVariable(RandomVariable):
    '''
    Class for defining a uniform random variable
    '''

    def __init__(self, min_val=0., max_val=0., max_moment=10):
        '''
        Initialize the uniform (gaussian) random variable with provided
        minimum/maximum values. Implementation wraps scipy.stats.uniform to get
        statistics/samples. Caches moments up to max_moment for speedup.
        '''

        if min_val >= max_val:
            raise ValueError("Minimum value must be less than maximum value")

        self._minimum_value = min_val
        self._range_size = max_val - min_val

        #set dimension (scalar), min/max to equal mean +/- 4stds
        self._dim = 1
        self._mins = [min_val]
        self._maxs = [max_val]

        #cache moments
        self.generate_moments(max_moment)
        self._max_moment = max_moment

    def get_dim(self):
        return self._dim

    def get_variance(self):
        '''
        Returns variance of uniform random variable
        '''
        return self._std**2.0

    def compute_moments(self, max_order):
        '''
        Returns moments up to order 'max_order' in numpy array.
        '''

        #TODO - calculate moments above max_moment on the fly & append to stored
        if max_order <= self._max_moment:
            moments = self._moments[:max_order]
        else:
            raise NotImplementedError("Moment above max_moment not handled yet")

        return moments


    def compute_CDF(self, x_grid):
        '''
        Returns numpy array of uniform CDF values at the points contained
        in x_grid
        '''

        return scipyuniform.cdf(x_grid, self._minimum_value, self._range_size)


    def compute_inv_CDF(self, x_grid):
        '''
        Returns np array of inverse uniform CDF values at pts in x_grid
        '''
        return scipyuniform.ppf(x_grid, self._minimum_value, self._range_size)


    def compute_pdf(self, x_grid):
        '''
        Returns numpy array of uniform pdf values at the points contained
        in x_grid
        '''
        return scipyuniform.pdf(x_grid, self._minimum_value, self._range_size)


    def draw_random_sample(self, sample_size):
        '''
        Draws random samples from the uniform random variable. Returns numpy
        array of length 'sample_size' containing these samples
        '''

        #Use scipy uniform rv to return shifted/scaled samples automatically
        return scipyuniform.rvs(self._minimum_value, self._range_size, 
                                sample_size)

    def generate_moments(self, max_moment):
        '''
        Calculate & store moments to retrieve more efficiently later
        '''

        self._moments = np.zeros((max_moment, 1))

        #Rely on scipy.stats to return non-central moment
        for i in range(max_moment):
            self._moments[i] = scipyuniform.moment(i+1, self._minimum_value, 
                                                   self._range_size)


