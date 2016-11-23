
import numpy as np


class ObjectiveFunction:
    '''
    Defines the objective function for optimizing SROM parameters. Calculates
    errors between the statistics of the SROM and the target random vector
    being model by it.
    Will create objective function for optimization library (e.g. scipy) that 
    essentially wraps this class's evaluate function
    '''

    def __init__(self, SROM, targetRV, obj_weights=None, error='mean',
                 max_moment=5):
        '''
        Initialize objective function. Pass in SROM & target random vector
        objects that have been previously initialized. Objective function
        calculates the errors between the statistics of this SROM and the 
        target random vector (these objects must have compute_moments,CDF,
        corr_mat functions defined). 

        inputs:
            -SROM - initialized SROM object
            -targetRV - initialized RandomVector object (either AnalyticRV or 
                SampleRV) with same dimension as SROM
            -obj_weights - array of floats defining the relative weight of the 
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. 
            -error - string 'mean' or 'max' defining how error is defined 
                between the statistics of the SROM & target
            -max_moment - int, max order to evaluate moment errors up to

        '''

        #TODO - make sure SROM & targetRV are properly initialized / have same d
        self._SROM = SROM
        self._target = targetRV 

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


    def evaluate(self, samples, probs, x_grid):
        '''
        Evaluates the objective function for the specified SROM samples & 
        probabilities. Calculates errrors in statistics between SROM/target

        #***** TODO - how to handle x_grid for evaluating CDF error? 
        #         - define in constructor with max moments?
        '''

        error = 0.0
 
        #SROM is now defined by the current values of samples/probs for stats
        self._SROM.set_params(samples, probs)

        if self._weights[0] > 0.0:
            moment_error = self.compute_moment_error()
            error += moment_error * self._weights[0]

        if self._weights[1] > 0.0:
            cdf_error = self.compute_CDF_error(x_grid)
            error += cdf_error * self._weights[1]

        if self._weights[2] > 0.0:
            corr_error = self.compute_correlation_error()
            error += corr_error * self._weights[2]

        return error

    def compute_moment_error(self):
        '''
        Calculate error in moments between SROM & target
        '''
        
        srom_moments = self._SROM.compute_moments(self._max_moment)
        target_moments = self._targetRV.compute_moments(self._max_moment)
        diffs = np.abs(srom_moments - target_moments)

        if self._metric == "MEAN":
            error = np.mean(diffs)
        else:
            error = np.max(diffs)

        return error


    def compute_CDF_error(self, x_grid):
        '''
        Calculate error in CDFs between SROM & target at pts in x_grid
        '''

        srom_cdfs = self._SROM.compute_CDF(x_grid)
        target_cdfs = self._targetRV.compute_CDF(x_grid)
        diffs = np.abs(srom_cdfs - target_cdfs)

        if self._metric == "MEAN":
            error = np.mean(diffs)
        else:
            error = np.max(diffs)

        return error


    def compute_correlation_error(self):
        '''
        Calculate error in correlation matrix between SROM & target
        ''' 

        srom_corr = self._SROM.compute_corr_mat()
        target_corr = self._targetRV.compute_corr_mat()
        diffs = np.abs(srom_corr - target_corr)

        if self._metric == "MEAN":
            error = np.mean(diffs)
        else:
            error = np.max(diffs)

        return error
