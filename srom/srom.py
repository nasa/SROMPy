'''
Define stochastic reduced order model (SROM) class
'''

import copy
import os
import numpy as np

from optimize import Optimizer

class SROM(object):
    '''
    Class defining the stochastic reduced order model (SROM). Used to calculate
    SROM statistics, optimize for parameters to match a given target random
    vector, and store to / load from file
    '''

    def __init__(self, size, dim):
        '''
        Initialize SROM w/ specified size for random vector of dimension dim

        m = SROM size (superscript), d = dimension (subscript);

        samples =  |  x^(1)_1, ..., x^(1)_d | ;    probs = | p^(1) |
                   |  ...    , ..., ...     |              | ...   |
                   |  x^(m)_1, ..., x^(m)_d |              | p^(m) |

        TODO - allow to be initialized with samples/probabilities?
        '''

        self._size = int(size)
        self._dim = int(dim)

        self._samples = None
        self._probs = None

    def set_params(self, samples, probs):
        '''
        Set defining SROM parameters - samples & corresponding probabilities

        inputs:
            -samples, numpy array, (SROM size) x (dim)
            -probs, numpy array, (SROM size) x 1

        '''

        #Handle 1 dimension case, adjust shape:
        if len(samples.shape) == 1:
            samples.shape = (len(samples), 1)

        #Verify dimensions of samples/probs
        (size, dim) = samples.shape

        if size != self._size and dim != self._dim:
            msg = "SROM samples have wrong dimension, must be (sromsize x dim)"
            raise ValueError(msg)

        if len(probs) != self._size:
            raise ValueError("SROM probs must have dim. equal to srom size")

        self._samples = copy.deepcopy(samples)
        self._probs = copy.deepcopy(probs.reshape((self._size, 1)))


    def compute_moments(self, max_order):
        '''
        Calculate SROM moments up to order 'max_order'. Returns a
        (max_order x dim) size numpy array with SROM moments for each dimension.
        Computes moments from 1,...,max_order
        '''

        #Make sure SROM has been properly initialized
        if self._samples is None or self._probs is None:
            raise ValueError("Must initalize SROM before computing moments")

        max_order = int(max_order)
        moments = np.zeros((max_order, self._dim))

        for q in range(max_order):

            #moment_q = sum_{k=1}^m p(k) * x(k)^q
            moment_q = np.zeros((1, self._dim))
            for k, sample in enumerate(self._samples):
                moment_q = moment_q + self._probs[k]* pow(sample, q+1)

            moments[q, :] = moment_q

        return moments

    def compute_CDF(self, x_grid):
        '''
        Evaluates the SROM CDF values for each dimension at each point in
        x_grid. x_grid can be a 1D array in which case the CDFs for each random
        vector dimension are evaluated at the same points, or it can be a
        (num_grid_pts x dim) array, specifying different points for each
        dimension - each dimension can have a different range of values but
        must have the same # of grid pts across it. Returns a (num_grid_pts x
        dim) array of corresponding CDF values at the grid points

        Note:
             -this is marginal CDFs along each dimension
             -increasing size of x_grid will slow this down a lot

        inputs:
            x_grid, (#grid pts x 1 ) or (#grid pts x dim) array of points to
                evaluate the SROM CDF at for case of the same grid for each dim
                or different grid, respectively.

        returns:
            cdf_vals, (#grid pts x dim) array of SROM CDF values at x_grid pts

        TODO - option for smoothed/differentiable SROM CDF approximation?
        '''

        #NOTE - should deep copy x_grid since were modifying?

        #Make sure SROM has been properly initialized
        if self._samples is None or self._probs is None:
            raise ValueError("Must initalize SROM before computing moments")

        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_pts, dim) = x_grid.shape

        #If only one grid was provided for multiple dims, repeat to generalize
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        CDF_vals = np.zeros((num_pts, self._dim))

        #Note probably a more efficient implementation - vectorize?:
        for d, grid in enumerate(x_grid.T):

            #CDF(x) = sum_{k=1}^m  1( sample^(k) < x) prob^(k)
            CDF_d = np.zeros((num_pts))
            for i, x_pt in enumerate(grid):
                for k, sample in enumerate(self._samples):
                    if sample[d] <= x_pt:
                        CDF_d[i] += self._probs[k]

            CDF_vals[:, d] = CDF_d

        return CDF_vals


    def compute_corr_mat(self):
        '''
        Returns the SROM correlation matrix as (dim x dim) numpy array

        srom_corr = sum_{k=1}^m [ x^(k) * (x^(k))^T ] * p^(k)

        '''

        #Make sure SROM has been properly initialized
        if self._samples is None or self._probs is None:
            raise ValueError("Must initalize SROM before computing moments")
        corr = np.zeros((self._dim, self._dim))

        for k, sample in enumerate(self._samples):
            corr = corr + np.outer(sample, sample) * self._probs[k]

        return corr

    def optimize(self, targetRV, weights=None, num_test_samples=500,
                 error='SSE', max_moment=5, cdf_grid_pts=100,
                 tol=None, options=None, method=None):
        '''
        Optimize for the SROM samples & probabilities to best match the
        target random vector statistics.

        inputs:
            -target - initialized RandomVector object (either AnalyticRV or
                SampleRV) that is being modeled with the SROM
            -weights - array of floats defining the relative weight of the
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. Default will give
                each term equal weight
            -num_test_samples - int,  # of randomly drawn sample sets of the
                target random vector to optimize probaiblities for during
                optimization process.
            -error -string 'sse', 'mean' or 'max' defining how error is
                quantified between the statistics of the SROM & target
            -max_moment - int, max order to evaluate moment errors up to
            -cdf_grid_pts - int, # pts to evaluate CDF errors on
            -tol, float, tolerance of scipy optimization algorithm TODO
            -options, dict, options for scipy optimization algorithm TODO
            -method, str, method specifying scipy optimization algorithm TODO

        '''

        #Use optimizer to form SROM objective func & gradient and minimize:
        opt = Optimizer(targetRV, self, weights, error, max_moment,
                        cdf_grid_pts)

        (samples, probs) = opt.get_optimal_params(num_test_samples, tol,
                                                  options, method)

        self.set_params(samples, probs)


    def save_params(self, outfile="srom_params.txt"):
        '''
        Write the SROM parameters to file.

        Stores array in following format (samples in each row with prob after)

                    comp_1  comp_2 ... comp_d   probability
        sample_1    x_1^(1) x_2^(1)             p^(1)
        sample_2    x_1^(2)
        ...
        sample_m    x_1^(m)            x_d^(m)  p^(m)

        '''

        #Make sure SROM has been properly initialized
        if self._samples is None or self._probs is None:
            raise ValueError("Must initalize SROM before saving to disk")

        srom_params = np.hstack((self._samples, self._probs))
        np.savetxt(outfile, srom_params)

    def load_params(self, infile="srom_params.txt"):
        '''
        Load SROM parameters from file.

        Assumes array in following format (samples in each row with prob after)

                    comp_1  comp_2 ... comp_d   probability
        sample_1    x_1^(1) x_2^(1)             p^(1)
        sample_2    x_1^(2)
        ...
        sample_m    x_1^(m)            x_d^(m)  p^(m)

        '''

        if not os.path.isfile(infile):
            raise IOError("SROM parameter input file does not exist: " + infile)

        srom_params = np.genfromtxt(infile)

        (size, dim) = srom_params.shape
        dim -= 1                        #Account for probabilities in last col

        if size != self._size and dim != self._dim:
            msg = "Dimension mismatch when loading SROM params from file"
            raise ValueError(msg)

        self._samples = srom_params[:, :-1]
        self._probs = srom_params[:, -1]

