import copy
import numpy as np

from target import RandomVector

#TODO - why the F do i need to do RV.RV??? Treating RV as the module not class
class AnalyticRV(RandomVector.RandomVector):
    '''
    Analytically-specified random vector whose components follow standard
    probability distributions (beta, gamma, normal, etc.). 
    '''

    def __init__(self, random_variables, correlation_matrix):
        '''
        Create analytic random vector with components that follow 
        standard probability distributions. Initialize using a list of 
        random variable objects that define each dimension as well as a 
        correlation matrix specifying the correlation structure between
        components

        inputs:
            random_variables - list of random variable objects with length
                               equal to the desired dimension of the analytic
                               random vector being created. Must have 
                               compute_moments and compute_CDF functions 
                               implemented.
            correlation_matrix - numpy array with size (dimension x dimension)
                               with correlation between each component. Must be
                               symmetric, square matrix.
        '''

        #TODO - error checking to make sure random variables are properly
        #initialized / constructed / have necessary functions / member variables
        # like _min / _max

        #Error checking on correlation matrix:
        self.verify_correlation_matrix(correlation_matrix)
        self._corr = copy.deepcopy(correlation_matrix)
 
        #Size of correlation matrix must match # random variable components:
        if self._corr.shape[0] != len(random_variables):
            raise ValueError("Dimension mismatch btwn corr mat & random vars")

        #Parent class (RandomVector) constructor, sets self._dim
#        super(AnalyticRV, self).__init__(len(random_variables))

        self._dim = len(random_variables)

        #Get min/max values for each component
        self._components = copy.deepcopy(random_variables)
        self._mins = np.zeros(self._dim)
        self._maxs = np.zeros(self._dim)
        
        for i in range(self._dim):
            self._mins[i] = self._components[i]._min
            self._maxs[i] = self._components[i]._max


    def verify_correlation_matrix(self, corr_matrix):
        '''
        Do error checking on the provided correlation matrix, e.g., is it 
        square? is it symmetric? 
        '''
        
        corr_matrix = np.array(corr_matrix)  #make sure it's an numpy array

        if len(corr_matrix.shape)==1:
            raise ValueError("Correlation matrix must be a 2D array!")
    
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square!")

        #Slick check for symmetry:
        if not np.allclose(corr_matrix, corr_matrix.T, 1e-6):
            raise ValueError("Correlation matrix must be symmetric!")

        #Make sure all entries are positive:
        if np.any(corr_matrix < 0):
            raise ValueError("Correlation matrix entries must be positive!")
 

    def compute_moments(self, max_order):
        '''
        Calculate random vector moments up to order max_moment based
        on samples. Moments from 1,...,max_order
        '''

        #Get moments up to max_order for each component of the vector
        moments = np.zeros((max_order, self._dim))
        for i in range(self._dim):
            moments[:, i] = self._components[i].get_moments(max_order)

        return moments


    def compute_CDF(self, x_grid):
        '''
        Evaluates the precomputed/stored CDFs at the specified x_grid values
        and returns. x_grid can be a 1D array in which case the CDFs for each 
        dimension are evaluated at the same points, or it can be a 
        (num_grid_pts x dim) array, specifying different points for each 
        dimension - each dimension can have a different range of values but
        must have the same # of grid pts across it. Returns a (num_grid_pts x
        dim) array of corresponding CDF values at the grid points
    
        '''

        #NOTE - should deep copy x_grid since were modifying?
        #1D random variable case
        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid),1))
        (num_pts, dim) = x_grid.shape

        #If only one grid was provided for multiple dims, repeat to generalize
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        CDF_vals = np.zeros((num_pts, self._dim))

        #Evaluate CDF interpolants on grid
        for d, grid in enumerate(x_grid.T):

            #Make sure grid values lie within max/min along each dimension
            grid[np.where(grid<self._mins[d])] = self._mins[d]
            grid[np.where(grid>self._maxs[d])] = self._maxs[d]

            CDF_vals[:, d] = self._components[d].compute_CDF(grid)

        return CDF_vals

    def compute_corr_mat(self):
        '''
        Returns the correlation matrix
        '''
        return self._corr

    def draw_random_samples(self, sample_size):
        '''
        TODO 
        '''
        return
               


