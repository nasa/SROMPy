'''
Class for defining a discrete random vector as a target to be matched with a 
SROM. Similar to the sample-based random vector, but in general will have
probabilities that are not equal. 
'''

import numpy as np

from src.target.RandomVector import RandomVector

class DiscreteRandomVector(RandomVector):
    '''
    Discrete random vector. Defines a target that can be matched with a SROM
    that is created from samples and corresponding probabilities. Implements
    basic discrete statistics (similar to those of an SROM). 

    :param samples: set of realizations/samples of the random vector
    :type samples: np array, size: (# samples x dim)
    :param probabilities: probabilties associated with each sample
    :type probabilities: np array, length = # samples
    :param max_moment: max. order moment to precompute and store
    :type max_moment: int
    '''

    def __init__(self, samples, probabilities, max_moment=10):

        #Check for 1D case (random variable)
        if len(samples.shape) == 1:
            samples = samples.reshape((len(samples), 1))

        self._validate_inputs(samples, probabilities)

        (num_samples, dim) = samples.shape
        self._samples = samples
        self._probabilities = probabilities
        self._max_moment = max_moment
        self._num_samples = num_samples
    
        #min/max sample values needed for SROM optimization
        self._mins = np.min(samples, axis=0)
        self._maxs = np.max(samples, axis=0)

        #Parent class (RandomVector) constructor, sets self._dim
        super(DiscreteRandomVector, self).__init__(dim)

        #Cache statistics so they can be returned quickly later
        self._precompute_moments()
        self._precompute_correlation_matrix()


    def compute_moments(self, max_order):
        '''
        Return precomputed moments up to specified order.

        :param max_order: Maximum order of moments to return
        :type max_order: int

        Returns (max_order x dim) size Numpy array with SROM moments for
        each dimension.
        '''

        #TODO - calculate moments above max_moment on the fly & append to stored
        if max_order <= self._max_moment:
            moments = self._moments[:max_order, :]
        else:
            raise NotImplementedError("Moment above max_moment not handled yet")

        return moments


    def compute_CDF(self, x_grid):
        '''
        Computes the marginal CDF values in each dimension.

        :param x_grid: Grid of points to compute CDF values on. If 1d array is
            provided, the same points are used to evaluate CDF in each
            dimension. If 2d array is provided, calculates CDF values on
            different points, but must have same # points for each dimension.
            Size is (# grid pts) x (dim) or (# grid pts) x (1).
        :type x_grid: Numpy array.

        Returns: Numpy array of CDF values at x_grid points. Size is (# grid
        pts) x (dim).

        Note:
            * Increasing the number of grid points can significantly slow
              down the SROM optimization problem.
            * Providing a 2d array for x_grid can specify a different range
              of values for each dimension, but must use the same number of pts.
        '''

        #Account for 1d random variable
        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_pts, dim) = x_grid.shape

        # If only one grid was provided for multiple dims, repeat to generalize
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        CDF_vals = np.zeros((num_pts, self._dim))

        # Vectorized indicator implementation for CDF
        # CDF(x) = sum_{k=1}^m  1( sample^(k) < x) prob^(k)
        for i, grid in enumerate(x_grid.T):
            for k, sample in enumerate(self._samples):
                indz = grid >= sample[i]
                CDF_vals[indz, i] += self._probabilities[k]

        return CDF_vals

    def compute_corr_mat(self):
        '''
        Returns precomputed correlation matrix.
        '''
        return self._corr_matrix

    def draw_random_sample(self, sample_size):
        '''
        Randomly draws a sample of this random vector.

        :param sample_size: number of samples to return
        :type sample_size: int

        sample_size must be smaller than total # of samples. For discrete
        random vector, we return a randomly selected # of samples
        '''

        if sample_size > self._num_samples:
            raise ValueError("Sample size can't be more than total # samples")

        #Generate random indices for samples array
        all_inds = np.arange(self._num_samples)
        random_inds = np.random.choice(all_inds, sample_size, replace=False)

        sample = self._samples[random_inds, :]

        return sample
    
    def _precompute_moments(self):
        '''
        Precomputes and stores moments and stores in moments member variable
        array
        '''
    
        self._moments = np.zeros((self._max_moment, self._dim))
        
        for order in range(self._max_moment):
        
            # moment_q = sum_{k=1}^m p(k) * x(k)^q
            moment_q = np.zeros((1, self._dim))
            for k, sample in enumerate(self._samples):
                moment_q = moment_q + self._probabilities[k] * \
                                      pow(sample, order+1)

            self._moments[order, :] = moment_q


    def _precompute_correlation_matrix(self):
        '''
        Precomputes and stores correlation matrix and stores in 
        "_corr_matrix" member variable array.
        '''
        corr = np.zeros((self._dim, self._dim))

        for k, sample in enumerate(self._samples):
            corr = corr + np.outer(sample, sample) * self._probabilities[k]

        self._corr_matrix = corr


    def _validate_inputs(self, samples, probabilities):
        '''
        Check shapes/sizes/types? of arrays for samples/probs and proper 
        probabilities
        '''

        (num_samples, dim) = samples.shape
        num_probs = len(probabilities)
    
        if num_samples != num_probs:
            msg = "Length of probability array must match # of samples"
            raise ValueError(msg)

        if num_samples < dim:
            msg = "Number of samples is less than dim, check sample array shape"
            raise ValueError(msg)

        if (probabilities < 0).any():
            raise ValueError("Probabilities cannot be negative!")

        if not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to one!")

