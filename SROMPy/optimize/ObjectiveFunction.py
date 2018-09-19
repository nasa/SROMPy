# Copyright 2018 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.

# The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed
# under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import numpy as np

from SROMPy.target import RandomVector
from SROMPy.target import RandomVariable


class ObjectiveFunction:
    '''
    Defines the objective function for optimizing SROM parameters. Calculates
    errors between the statistics of the SROM and the target random vector
    being model by it.
    Will create objective function for optimization library (e.g. scipy) that 
    essentially wraps this class's evaluate function
    '''

    def __init__(self,
                 srom,
                 target,
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

        # Test target
        if not (isinstance(target, RandomVector) or isinstance(target, RandomVariable)):
            raise TypeError("target must inherit from RandomVector or RandomVariable.")

        # Test srom
        from SROMPy.srom import SROM
        if not isinstance(srom, SROM):
            raise TypeError("srom must be of type SROM.")

        # Ensure srom and target have same dimensions if target is a RandomVector.
        if isinstance(target, RandomVector):

            if target.get_dim() != srom.get_dim():
                raise ValueError("target and srom must have same dimensions.")

        self._SROM = srom
        self._target = target

        # Generate grids for evaluating CDFs based on target RV's range
        self.generate_cdf_grids(cdf_grid_pts)

        # Test obj_weights
        if obj_weights is not None:

            if isinstance(obj_weights, list):
                obj_weights = np.array(obj_weights)

            if not isinstance(obj_weights, np.ndarray):
                raise TypeError("obj_weights must be of type ndarray.")

            if len(obj_weights.shape) != 1:
                raise ValueError("obj_weights must be a one dimensional array.")

            if obj_weights.shape[0] != 3:
                raise ValueError("obj_weights must have exactly three elements.")

            if np.min(obj_weights) < 0.:
                raise ValueError("obj_weights cannot have values less than zero.")
            self._weights = obj_weights
        else:
            self._weights = np.ones((3,))        

        # Test error function name.
        if not isinstance(error, str):
            raise TypeError("error must be a string, either 'MEAN', 'MAX', or 'SSE'.")

        if error.upper() not in ["MEAN", "MAX", "SSE"]:
            raise ValueError("error must be either 'mean', 'max', or 'SSE'.")

        # Test max_moment.
        if not isinstance(max_moment, int):
            raise TypeError("max_moment must be a positive integer.")

        if max_moment < 1:
            raise ValueError("max_moment must be a positive integer.")

        # Test cdf_grid_pts.
        if not isinstance(cdf_grid_pts, int):
            raise TypeError("cf_grid_pts must be a positive integer.")

        if cdf_grid_pts < 1:
            raise ValueError("cdf_grid_pts must be a positive integer.")

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
 
         #SROM is now defined by the current values of samples/probs for stats.
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

        # Reshape to 2D if returned as 1D for scalar RV.
        if len(target_moments.shape)==1:
            target_moments = target_moments.reshape((self._max_moment, 1))

        # Prevent divide by zero.
        zeroinds = np.where(np.abs(target_moments) <= 1e-12)[0]
        target_moments[zeroinds] = 1.0

        # Squared relative difference:
        if self._metric == "SSE":
            rel_diffs = ((srom_moments-target_moments)/target_moments)**2.0
            error = 0.5*np.sum(rel_diffs)
        # Max absolute value:
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

        # Check for 0 cdf vals to prevent divide by zero.
        nonzeroind = np.where(target_cdfs[:, 0] > 0)[0]
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

        # Neglect for 1D random variable:
        if self._target.get_dim() == 1:
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
        
        self._x_grid = np.zeros((cdf_grid_pts, self._target.get_dim()))

        for i in range(self._target.get_dim()):
            grid = np.linspace(self._target._mins[i], 
                               self._target._maxs[i],
                               cdf_grid_pts)
            self._x_grid[:, i] = grid

