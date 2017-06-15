
import numpy as np


class Gradient:
    '''
    Defines gradients of objective function w/ respect to srom parameters 
    for optimizing SROM parameters. Will be used to pass derivative info
    to scipy optimization library for faster minimization.
    '''

    def __init__(self, SROM, targetRV, obj_weights=None, error='mean',
                 max_moment=5, cdf_grid_pts=100):
        '''
        Initialize SROM obj fun gradient. Pass in SROM & target random vector
        objects that have been previously initialized. 

        inputs:
            -SROM - initialized SROM object
            -targetRV - initialized RandomVector object (either AnalyticRV or 
                SampleRV) with same dimension as SROM
            -obj_weights - array of floats defining the relative weight of the 
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. 
            -max_moment - int, max order to evaluate moment errors up to
            -cdf_grid_pts - int, # pts to evaluate CDF errors on
        '''

        #NOTE - gradients won't make sense for MAX error metric NOTE NOTE NOTE
        #Need to decide if objective function terms are normalized by true val

        #Error checking/handling should have already been done by obj fun prior
        self._SROM = SROM
        self._target = targetRV

        #Generate grids for evaluating CDFs based on target RV's range
        self.generate_cdf_grids(cdf_grid_pts)

        if obj_weights is not None:
            if len(obj_weights) != 3:
                raise ValueError("obj_weights must have length 3!")
            self._weights = obj_weights
        else:
            self._weights = np.ones((3,))

        if error.upper() not in ["MEAN", "MAX"]:
            raise ValueError("error must be either 'mean' or 'max'")
        self._metric = error.upper()

        self._max_moment = max_moment

    def evaluate(self, samples, probs):
        '''
        Evaluates gradient (for probability only)
        Just calls gradient_wrt_probs() for now
        '''
        return self.gradient_wrt_probs(samples, probs)


    def gradient_wrt_probs(self, samples, probs):
        '''
        Returns gradient vector w/ derivative of obj function w.r.t. SROM
        probabilities (m x 1 array)
        '''
    
        sromsize = self._SROM._size
        grad = np.zeros((sromsize,1))

        for srom_ind in range(sromsize):

            #d(CDF_error)/dp_i :
            if self._weights[0] > 0:
                cdf_grad = self.CDF_wrt_prob(samples, probs, srom_ind)
            else:
                cdf_grad = 0.0
            
            #d(moment_error)/dp_i: 
            if self._weights[1] > 0:
                moment_grad = self.moment_wrt_prob(samples, probs, srom_ind)
            else:
                moment_grad = 0.0
            
            #d(corr_error)/dp_i
            if self._weights[2] > 0:
                corr_grad = self.corr_wrt_prob(samples, probs, srom_ind)
            else:
                corr_grad = 0.0
            
            #Sum contribution to gradient from each error term 
            grad[srom_ind] = (self._weights[0]*cdf_grad + 
                              self._weights[1]*moment_grad + 
                              self._weights[2]*corr_grad)  

        grad = grad.flatten()
        
        return grad
    
    def CDF_wrt_prob(self, samples, probs, srom_ind):
        '''
        Gradient of CDF error term with respect to probability (for srom_ind)
        
        -Expression - the "erf" term of the gradient from the SROM paper 
        becomes an indicator function when smooth CDF is not used
        '''
        
        (size, dim) = samples.shape

        srom_cdfs = self._SROM.compute_CDF(self._x_grid)
        target_cdfs = self._target.compute_CDF(self._x_grid)
        diffs = (srom_cdfs - target_cdfs)/target_cdfs**2.0

        #TODO -vectorize this? Get indicator array & do pt wise multiply
        grad = 0
        samples_k = samples[srom_ind,:]
        for j, grid_pt in enumerate(self._x_grid): #1 grid pt in all dims
            for i, pt in enumerate(grid_pt): #ith dim of jth grid pt
                if samples_k[i] <= pt: #indicator x_srom^k_i <= x_grid_ij
                    grad += diffs[j,i]

        #Take into accoutn averaging:
        factor = diffs.shape[0] * diffs.shape[1]
        grad *= (1. / factor)
        return grad


    def moment_wrt_prob(self, samples, probs, srom_ind):
        '''
        Gradient of moment error term with respect to probability (for srom_ind)
        '''
        
        (size, dim) = samples.shape
        
        srom_moments = self._SROM.compute_moments(self._max_moment)
        target_moments = self._target.compute_moments(self._max_moment)
        diffs = (srom_moments - target_moments)/target_moments**2.0

        grad_sum = 0.0
        sample_k = samples[srom_ind,:]

        for i in range(dim):

            for q in range(self._max_moment):
                #grad_i += (1./srom_moments[k,i]) * ... 
                grad_sum += diffs[q,i]*sample_k[i]**q

        factor = diffs.shape[0] * diffs.shape[1]
        grad_sum *= (1. / factor)

        return grad_sum


    def corr_wrt_prob(self, samples, probs, srom_ind):
        '''
        Gradient of corr. error term with respect to probability (for srom_ind)
        '''

        (size, dim) = samples.shape
    
        #Correlation irrelevant for 1D
        if dim == 1:
            return 0.0
        
        srom_corr = self._SROM.compute_corr_mat()
        target_corr = self._target.compute_corr_mat()
        diffs = (srom_corr - target_corr) / target_corr**2.0

        grad_sum = 0.0

        sample_k = samples[srom_ind, :]

        for i in range(dim):
            for j in range(dim):
                #grad_sum +=  (1/true_corrleation^2)  ... if normalized obj fun
                grad_sum += diffs[i,j] * sample_k[i] * sample_k[j]
    
        factor = diffs.shape[0] * diffs.shape[1]
        grad_sum *= (1. / factor)

        return grad_sum

    def generate_cdf_grids(self, cdf_grid_pts):
        '''
        Generate numerical grids for evaluating the CDF errors based on the 
        range of the target random vector. Create x_grid member variable with
        cdf_grid_pts along each dimension of the random vector.
        '''

        self._x_grid = np.zeros((cdf_grid_pts, self._target._dim))

        for i in range(self._target._dim):
            grid = np.linspace(self._target._mins[i],
                               self._target._maxs[i],
                               cdf_grid_pts)
            self._x_grid[:,i] = grid
