'''
Class for defining a sample-based random vector with empirical estimators
'''

import numpy as np
from scipy import interpolate

from SROMPy.target import RandomVector


class SampleRandomVector(RandomVector):
    '''
    Sample-based random vector. Defines a target random vector to match with
    an SROM based on a set of realizations of that random vector. Implements
    basic statistics to use in SROM optimization and comparisons.

    :param samples: set of realizations/samples of the random vector
    :type samples: np array, size: (# samples x dim)
    :param max_moment: max. order moment to precompute and store
    :type max_moment: int
    '''

    def __init__(self, samples, max_moment=10):
        '''
        Initialize SampleRandomVector with an array of samples of the random vector.
        Must be an array of size (# samples x dim). Statistics of the
        SampleRandomVector are precomputed during initialization - max_moment is the
        maximum moment order to compute & store for later use. If higher
        moments are anticipated, this can be increased (or visa versa)
        '''

        #Check for 1D case (random variable)
        if len(samples.shape) == 1:
            samples = samples.reshape((len(samples), 1))

        (num_samples, dim) = samples.shape

        if dim > num_samples:
            msg = "Dimension is greater than # samples! Check samples array"
            raise ValueError(msg)

        #Parent class (RandomVector) constructor, sets self._dim
        super(SampleRandomVector, self).__init__(dim)

        self._num_samples = num_samples
        self._samples = samples
        self._max_moment = max_moment

        #Precompute & store statistics so they can be returned quickly later
        self.generate_statistics(max_moment)

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
        Evaluates the precomputed/stored CDFs at the specified x_grid values
        and returns. 

        :param x_grid: Grid of points to compute CDF values on. If 1d array is
            provided, the same points are used to evaluate CDF in each
            dimension. If 2d array is provided, calculates CDF values on
            different points, but must have same # points for each dimension.
            Size is (# grid pts) x (dim) or (# grid pts) x (1).
        :type x_grid: Numpy array.

        Returns: Numpy array of CDF values at x_grid points. Size is (# grid
        pts) x (dim).
        '''

        #1D random variable case
        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_pts, dim) = x_grid.shape

        #If only one grid was provided for multiple dims, repeat to generalize
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        CDF_vals = np.zeros((num_pts, self._dim))

        #Evaluate CDF interpolants on grid
        for d, grid in enumerate(x_grid.T):

            #Make sure grid values lie within max/min along each dimension
            grid[np.where(grid < self._mins[d])] = self._mins[d]
            grid[np.where(grid > self._maxs[d])] = self._maxs[d]

            CDF_d = self._CDFs[d](grid)
            CDF_vals[:, d] = CDF_d

        return CDF_vals

    def compute_corr_mat(self):
        '''
        Returns precomputed correlation matrix.
        '''
        return self._corr

    def draw_random_sample(self, sample_size):
        '''
        Randomly draws a sample of this random vector.

        :param sample_size: number of samples to return
        :type sample_size: int

        sample_size must be smaller than total # of samples. For sample-based
        random vector, we return a randomly selected # of samples
        '''

        if sample_size > self._num_samples:
            raise ValueError("Sample size can't be more than total # samples")

        #Generate random indices for samples array
        all_inds = np.arange(self._num_samples)
        random_inds = np.random.choice(all_inds, sample_size, replace=False)

        sample = self._samples[random_inds, :]

        return sample


    #--------------Helper initialization methods---------------------------

    def get_plot_CDFs(self):
        '''
        Get CDF values for plotting (without using interpolant) - returns
        tuple with xgrid & CDF values arrays
        '''

        x_grid = np.zeros((self._num_samples, self._dim))
        CDF_vals = np.zeros((self._num_samples, self._dim))

        for i, samples_i in enumerate(self._samples.T):
            #Generate empirical CDF:
            sorted_i = np.sort(samples_i)
            cdf_vals = np.arange(1, len(sorted_i) + 1)/float(len(sorted_i))
            x_grid[:, i] = sorted_i
            CDF_vals[:, i] = cdf_vals

        return (x_grid, CDF_vals)

    def generate_statistics(self, max_moment):
        '''
        Precompute & store moments, CDFs, correlation matrix of the samples
        so that they can be returned quickly later
        '''

        self.generate_moments(max_moment)
        self.generate_CDFs()
        self.generate_correlation()

    def generate_moments(self, max_moment):
        '''
        Calculate & store random vector moments up to order max_moment based
        on samples. Moments from 1,...,max_order
        '''

        self._moments = np.zeros((max_moment, self._dim))

        factor = (1./float(self._num_samples))
        for q in range(0, max_moment):

            moment_q = np.zeros((1, self._dim))
            for sample in self._samples:
                moment_q += factor*np.power(sample, q+1)

            self._moments[q, :] = moment_q

    def generate_CDFs(self):
        '''
        Calculate & store marginal CDFs for each dimension of the random vector.
        Stores a linear interpolator of the CDF for each dim
        Uses trick from :   http://stackoverflow.com/questions/3209362/
                            %20how-to-plot-empirical-cdf-in-matplotlib-in-python
        to calculate CDF from samples
        '''

        self._CDFs = []

        #Need to store max/min samples in each dimension to prevent out of
        #bounds values in the interpolators later
        self._mins = []
        self._maxs = []

        #Get all samples of the i^th dimension at a time to generate CDF
        #NOTE - does iterating over / sorting happen in place? Need deep copy?
        for samples_i in self._samples.T:

            #Generate/store interpolant for empirical CDF:
            sorted_i = np.sort(samples_i)
            cdf_vals = np.arange(1, len(sorted_i) + 1)/float(len(sorted_i))
            cdf_func = interpolate.interp1d(sorted_i, cdf_vals)
            self._CDFs.append(cdf_func)

            self._mins.append(sorted_i[0])
            self._maxs.append(sorted_i[-1])


    def generate_correlation(self):
        '''
        Calculates and stores sample-based correlation matrix for random vector
        '''

        #TODO - find faster numpy/scipy function
        self._corr = np.zeros((self._dim, self._dim))

        factor = (1./float(self._num_samples))
        for sample in self._samples:
            self._corr = self._corr + factor*np.outer(sample, sample)





