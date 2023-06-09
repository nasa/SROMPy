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
from SROMPy.target.RandomEntity import RandomEntity


class ObjectiveFunction:
    """
    Defines the objective function for optimizing SROM parameters. Calculates
    errors between the statistics of the SROM and the target random vector
    being model by it.
    Will create objective function for optimization library (e.g. scipy) that 
    essentially wraps this class's evaluate function
    """

    def __init__(self, srom, target, obj_weights=None, error='mean',
                 max_moment=5, num_cdf_grid_points=100, joint_opt=False):
        """
        Initialize objective function. Pass in SROM & target random vector
        objects that have been previously initialized. Objective function
        calculates the errors between the statistics of this SROM and the 
        target random vector (these objects must have compute_moments,CDF,
        corr_mat functions defined). 

        inputs:
            -SROM - initialized SROM object
            -targetRV - initialized RandomVector object (either
                AnalyticRandomVector or SampleRandomVector) with same
                dimension as SROM
            -obj_weights - array of floats defining the relative weight of the 
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. Default is equal 
                weights ([1.0,1.0,1.0])
            -error - string 'mean','max', or 'sse' defining how error is defined
                between the statistics of the SROM & target
            -max_moment - int, max order to evaluate moment errors up to
            -num_cdf_grid_points - int, # pts to evaluate CDF errors on

        """

        self.__test_init_params(srom, target, obj_weights, error,
                                max_moment, num_cdf_grid_points)

        self._srom = srom
        self._target = target
        self._x_grid = None

        # Joint optimization
        self._joint_opt = joint_opt

        # Generate grids for evaluating CDFs based on target RV's range
        self.generate_cdf_grids(num_cdf_grid_points)

        self._metric = error.upper()

        self._max_moment = max_moment

    def get_moment_error(self, samples, probabilities):
        """
        Returns moment error for given samples & probabilities
        """

        self._srom.set_params(samples, probabilities)
        return self.compute_moment_error()

    def get_cdf_error(self, samples, probabilities):
        """
        Returns CDF error for given samples & probabilities
        """

        self._srom.set_params(samples, probabilities)
        return self.compute_cdf_error()

    def get_corr_error(self, samples, probabilities):
        """
        Returns correlation error for given samples & probabilities
        """

        self._srom.set_params(samples, probabilities)
        return self.compute_correlation_error()

    def evaluate(self, samples, probabilities):
        """
        Evaluates the objective function for the specified SROM samples & 
        probabilities. Calculates errrors in statistics between SROM/target
        """

        samples = self.check_bounds(samples.flatten(), self.get_param_bounds(joint_opt=self._joint_opt))
        error = 0.0
 
        # SROM is by the current values of samples/probabilities for stats.
        self._srom.set_params(samples, probabilities)

        if self._weights[0] > 0.0:
            cdf_error = self.compute_cdf_error()
            error += cdf_error * self._weights[0]

        if self._weights[1] > 0.0:
            moment_error = self.compute_moment_error()
            error += moment_error * self._weights[1]

        if self._weights[2] > 0.0:
            corr_error = self.compute_correlation_error()
            error += corr_error * self._weights[2]

        return error

    def get_param_bounds(self, joint_opt):
        """
        Get the bounds on parameters for SROM optimization problem. If doing
        joint optimization, need bounds for both samples & probabilities. If
        not, just need trivial bounds on probabilities
        """

        if not joint_opt:
            bounds = [(0.0, 1.0)] * self._srom.size
        else:
            bounds = list(zip(self._target.mins, self._target.maxs)) * self._srom.size + [(0.0, 1.0)] * self._srom.size

        return bounds

    def check_bounds(self, samples, bounds):
        for i in range(self._srom.size):
            x = samples[i]
            if np.any(x <= bounds[i][0]) or np.any(x >= bounds[i][1]):
                samples[i] = np.clip(x, bounds[i][0] + 1e-2, bounds[i][1] - 1e-2)
        return samples.reshape(self._srom.size, self._srom.dim)

    def compute_moment_error(self):
        """
        Calculate error in moments between SROM & target
        """
        
        srom_moments = self._srom.compute_moments(self._max_moment)
        target_moments = self._target.compute_moments(self._max_moment)

        # Reshape to 2D if returned as 1D for scalar RV.
        if len(target_moments.shape) == 1:
            target_moments = target_moments.reshape((self._max_moment, 1))

        # Prevent divide by zero.
        zero_indices = np.where(np.abs(target_moments) <= 1e-12)[0]
        target_moments[zero_indices] = 1.0

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

    def compute_cdf_error(self):
        """
        Calculate error in CDFs between SROM & target at pts in x_grid
        """

        srom_cdfs = self._srom.compute_cdf(self._x_grid)
        target_cdfs = self._target.compute_cdf(self._x_grid)

        # Check for 0 cdf values to prevent divide by zero.
        nonzero_indices = np.where(target_cdfs[:, 0] > 0)[0]
        srom_cdfs = srom_cdfs[nonzero_indices, :]
        target_cdfs = target_cdfs[nonzero_indices, :]

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
        """
        Calculate error in correlation matrix between SROM & target
        """ 

        # Neglect for 1D random variable:
        if self._target.dim == 1:
            return 0.0

        srom_corr = self._srom.compute_corr_mat()
        target_corr = self._target.compute_correlation_matrix()

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

    def generate_cdf_grids(self, num_cdf_grid_points):
        """
        Generate numerical grids for evaluating the CDF errors based on the 
        range of the target random vector. Create x_grid member variable with
        num_cdf_grid_points along each dimension of the random vector.
        """
        
        self._x_grid = np.zeros((num_cdf_grid_points, self._target.dim))

        for i in range(self._target.dim):
            grid = np.linspace(self._target.mins[i],
                               self._target.maxs[i],
                               num_cdf_grid_points)
            self._x_grid[:, i] = grid

    def __test_init_params(self, srom, target, obj_weights, error, max_moment,
                           num_cdf_grid_points):
        """
        Due to the large numbers of parameters passed into __init__() that
        need to be tested, the testing is done in this utility function
        instead of __init__().
        """

        # Test target.
        if not (isinstance(target, RandomEntity)):
            raise TypeError("target must inherit from RandomEntity.")

        # Test srom.
        from SROMPy.srom import SROM
        if not isinstance(srom, SROM):
            raise TypeError("srom must be of type SROM.")

        # Ensure srom and target have same dimensions if target is RandomVector.
        if isinstance(target, RandomVector):

            if target.dim != srom.dim:
                raise ValueError("target and srom must have same dimensions.")

        # Test obj_weights.
        if obj_weights is not None:

            if isinstance(obj_weights, list):
                obj_weights = np.array(obj_weights)

            if not isinstance(obj_weights, np.ndarray):
                raise TypeError("obj_weights must be of type ndarray or list.")

            if len(obj_weights.shape) != 1:
                raise ValueError("obj_weights must be a one dimensional array.")

            if obj_weights.shape[0] != 3:
                raise ValueError("obj_weights must have exactly 3 elements.")

            if np.min(obj_weights) < 0.:
                raise ValueError("obj_weights cannot be less than zero.")
            self._weights = obj_weights
        else:
            self._weights = np.ones((3,))

        # Test error function name.
        if not isinstance(error, str):
            raise TypeError("error must be a string: 'MEAN', 'MAX', or 'SSE'.")

        if error.upper() not in ["MEAN", "MAX", "SSE"]:
            raise ValueError("error must be either 'mean', 'max', or 'SSE'.")

        # Test max_moment.
        if not isinstance(max_moment, int):
            raise TypeError("max_moment must be a positive integer.")

        if max_moment < 1:
            raise ValueError("max_moment must be a positive integer.")

        # Test num_cdf_grid_points.
        if not isinstance(num_cdf_grid_points, int):
            raise TypeError("cf_grid_pts must be a positive integer.")

        if num_cdf_grid_points < 1:
            raise ValueError("num_cdf_grid_points must be a positive integer.")
