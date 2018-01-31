
import numpy as np
from scipy.stats import beta as scipybeta


class BetaRandomVariable(object):

    def __init__(self, alpha, beta, shift=0, scale=1):
        '''
        Initialize the beta random variable with the standard alpha and beta
        shape parameters (follows convention for a & b in numpy.random.beta).
        Optionally specify shift & scale parameters to translate and scale the
        random variable, e.g.:
            new_beta = shift + scale * standard_beta.
    
        If one wants to specify a beta random variable to match a given 
        support (min, max), mean, and variance, use the static method
        get_beta_shape_params() to convert to inputs for this constructor.
    
        Implementation wraps scipy.stats.beta to get statistics/samples.
        '''
     
        if alpha < 0:
            raise ValueError("Alpha shape param must be non-negative")
        if beta < 0:
            raise ValueError("Beta shape param must be non-negative")
        if scale <= 0: 
            raise ValueErorr("Scale param must be positive")
    
        self._alpha = alpha
        self._beta = beta
        self._shift = shift
        self._scale = scale
        self._dim = 1

    @staticmethod
    def get_beta_shape_params(min_val, max_val, mean, var):
        '''
        Returns the beta shape parameters (alpha, beta) and the shift/scale
        parameters that produce a beta random variable with the specified
        minimum value, maximum value, mean, and variance. Can be called prior
        to initialization of this class if only this info is known about the
        random variable being modeled. 
        Returns a list of length 4 ordered [alpha, beta, shift, scale]
        '''
    
        #Cast to make sure we have floats for calculations
        min_val = float(min_val)
        max_val = float(max_val)
        mean_val = float(mean)
        var = float(var)

        #Scale mean/variance to lie in [0,1] for standard beta distribution
        mean_std = (mean - min_val)/(max_val - min_val)
        var_std = (1. / (max_val - min_val))**2.0 * var

        #Get shape params based on scaled mean/variance:
        alpha = mean_std*( mean_std*(1. - mean_std)/ var_std - 1.)
        beta = (mean_std*(1 - mean_std)/var_std - 1) - alpha
        shift = min_val
        scale = max_val - min_val

        return [alpha, beta, shift, scale]

    def compute_moments(self, max_order):
        '''
        Returns moments up to order 'max_order' in numpy array.
        '''

        #Rely on scipy.stats to return non-central moment
        moments = np.zeros(max_order)
        for i in range(max_order):
            moments[i] = scipybeta.moment(i+1, self._alpha, self._beta, 
                                          self._shift, self._scale)
        return moments


    def compute_CDF(self, x_grid):
        '''
        Returns numpy array of beta CDF values at the points contained in x_grid
        '''

        return scipybeta.cdf(x_grid, self._alpha, self._beta, self._shift,
                             self._scale)

    def compute_pdf(self, x_grid):
        '''
        Returns numpy array of beta pdf values at the points contained in x_grid
        '''
        return scipybeta.pdf(x_grid, self._alpha, self._beta, self._shift,
                             self._scale)

    def draw_random_sample(self, sample_size):
        '''
        Draws random samples from the beta random variable. Returns numpy
        array of length 'sample_size' containing these samples
        '''

        #Use scipy beta rv to return shifted/scaled samples automatically
        return scipybeta.rvs(self._alpha, self._beta, self._shift, self._scale,
                             sample_size)
        
