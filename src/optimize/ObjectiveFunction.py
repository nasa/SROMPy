
import numpy as np


class ObjectiveFunction:
    '''
    Defines the objective function for optimizing SROM parameters. Calculates
    errors between the statistics of the SROM and the target random vector
    being model by it.
    Will create objective function for optimization library (e.g. scipy) that 
    essentially wraps this class's evaluate function
    '''

    def __init__(self,
                 SROM,
                 targetRV,
                 obj_weights=None,
                 error='mean',
                 max_moment=5,
                 cdf_grid_pts=100):
        '''
        Initialize objective function. Pass in SROM & target random vector
        objects that have been previously initialized. Objective function
        calculates the errors between the statistics of this SROM and the 
        target random vector (these objects must have compute_moments,CDF,
        corr_mat functions defined). 

        inputs:
            -SROM - initialized SROM object
            -targetRV - initialized RandomVector object (either AnalyticRandomVector or
                SampleRandomVector) with same dimension as SROM
            -obj_weights - array of floats defining the relative weight of the 
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. Default is equal 
                weights ([1.0,1.0,1.0])
            -error - string 'mean','max', or 'sse' defining how error is defined
                between the statistics of the SROM & target
            -max_moment - int, max order to evaluate moment errors up to
            -cdf_grid_pts - int, # pts to evaluate CDF errors on

        '''

        #TODO - make sure SROM & targetRV are properly initialized / have same d
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

        if error.upper() not in ["MEAN", "MAX", "SSE"]:
            raise ValueError("error must be either 'mean' or 'max'")
        self._metric = error.upper()

        self._max_moment = max_moment

    def get_moment_error(self, samples, probs):
        '''
        Returns moment error for given samples & probs
        '''
        self._SROM.set_params(samples, probs)
        return self.compute_moment_error()


    def get_cdf_error(self, samples, probs):
        '''
        Returns CDF error for given samples & probs
        '''
        self._SROM.set_params(samples, probs)
        return self.compute_CDF_error()


    def get_corr_error(self, samples, probs):
        '''
        Returns correlation error for given samples & probs
        '''
        self._SROM.set_params(samples, probs)
        return self.compute_correlation_error()


    def evaluate(self, samples, probs):
        '''
        Evaluates the objective function for the specified SROM samples & 
        probabilities. Calculates errrors in statistics between SROM/target

        '''

        error = 0.0
 
        #SROM is now defined by the current values of samples/probs for stats
        self._SROM.set_params(samples, probs)

        if self._weights[0] > 0.0:
            cdf_error = self.compute_CDF_error()
            error += cdf_error * self._weights[0]

        if self._weights[1] > 0.0:
            moment_error = self.compute_moment_error()
            error += moment_error * self._weights[1]

        if self._weights[2] > 0.0:
            corr_error = self.compute_correlation_error()
            error += corr_error * self._weights[2]

        return error

    def compute_moment_error(self):
        '''
        Calculate error in moments between SROM & target
        '''
        
        srom_moments = self._SROM.compute_moments(self._max_moment)
        target_moments = self._target.compute_moments(self._max_moment)

        #Reshape to 2D if returned as 1D for scalar RV
        if len(target_moments.shape)==1:
            target_moments = target_moments.reshape((self._max_moment, 1))

        #Prevent divide by zero
        zeroinds = np.where(np.abs(target_moments) <= 1e-12)[0]
        target_moments[zeroinds] = 1.0

        #Squared relative difference:
        if self._metric == "SSE":
            rel_diffs = ((srom_moments-target_moments)/target_moments)**2.0
            error = 0.5*np.sum(rel_diffs)
        #Max absolute value:
        elif self._metric == "MAX":
            diffs = np.abs(srom_moments - target_moments)
            error = np.max(diffs)
        elif self._metric == "MEAN":    
            diffs = np.abs(srom_moments - target_moments)
            error = np.mean(diffs)
        else:
            raise ValueError("Invalid error metric")

        return error


    def compute_CDF_error(self):
        '''
        Calculate error in CDFs between SROM & target at pts in x_grid
        '''

        srom_cdfs = self._SROM.compute_CDF(self._x_grid)
        target_cdfs = self._target.compute_CDF(self._x_grid)

        #Check for 0 cdf vals to prevent divide by zero
        nonzeroind = np.where(target_cdfs[:,0]>0)[0]
        srom_cdfs = srom_cdfs[nonzeroind, :]
        target_cdfs = target_cdfs[nonzeroind, :]

        if self._metric == "SSE":
            squared_diffs = (srom_cdfs - target_cdfs)**2.0
            rel_diffs = squared_diffs / target_cdfs**2.0
            error = 0.5*np.sum(rel_diffs)
        elif self._metric == "MAX":
            diffs = np.abs(srom_cdfs - target_cdfs)
            error = np.max(diffs)
        elif self._metric == "MEAN":    
            diffs = np.abs(srom_cdfs - target_cdfs)
            error = np.mean(diffs)
        else:
            raise ValueError("Invalid error metric")

        return error


    def compute_correlation_error(self):
        '''
        Calculate error in correlation matrix between SROM & target
        ''' 

        #Neglect for 1D random variable:
        if self._target._dim == 1:
            return 0.0

        srom_corr = self._SROM.compute_corr_mat()
        target_corr = self._target.compute_corr_mat()

        if self._metric == "SSE":
            squared_diffs = (srom_corr - target_corr)**2.0
            rel_diffs = squared_diffs / target_corr**2.0
            error = 0.5*np.sum(rel_diffs)
        elif self._metric == "MAX":
            diffs = np.abs(srom_corr - target_corr)
            error = np.max(diffs)
        elif self._metric == "MEAN":    
            diffs = np.abs(srom_corr - target_corr)
            error = np.mean(diffs)
        else:
            raise ValueError("Invalid error metric")

        return error

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

