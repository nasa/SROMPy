'''
Define stochastic reduced order model (SROM) class
'''

import numpy as np


class SROM:

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

        #TODO - do I need a deep copy? 
        self._samples = samples
        self._probs = probs


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
    
        for q in range(0,max_order):

            #moment_q = sum_{k=1}^m p(k) * x(k)^q
            moment_q = np.zeros((1, self._dim))
            for k, sample in enumerate(self._samples):
                moment_q = moment_q + self._probs[k]* np.power(sample, q+1)
            
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
            x_grid = x_grid.reshape((len(x_grid),1))
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

            CDF_vals[:,d] = CDF_d

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


