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


class Gradient:
    """
    Defines gradients of objective function w/ respect to srom parameters 
    for optimizing SROM parameters. Will be used to pass derivative info
    to scipy optimization library for faster minimization.
    """

    def __init__(self, srom, target_random_variable, obj_weights=None,
                 error='mean', max_moment=5, cdf_grid_pts=100, scale=None, joint_opt=False):
        """
        Initialize SROM obj fun gradient. Pass in SROM & target random vector
        objects that have been previously initialized. 

        inputs:
            -SROM - initialized SROM object
            -targetRV - initialized RandomVector object (either
            AnalyticRandomVector or SampleRandomVector) with same dimension as
            SROM
            -obj_weights - array of floats defining the relative weight of the 
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. 
            -max_moment - int, max order to evaluate moment errors up to
            -cdf_grid_pts - int, # pts to evaluate CDF errors on
        """

        # NOTE - gradients won't make sense for MAX error metric

        self.__check_init_parameters(obj_weights, error, scale)
        # Error checking/handling should have already been done by obj fun prior
        self.srom = srom
        self._target = target_random_variable
        self._x_grid = None

        # Scale for error function when using SSE for smooth derivative
        self._scale = scale
        self._joint_opt = joint_opt

        # Generate grids for evaluating CDFs based on target RV's range
        self._generate_cdf_grids(cdf_grid_pts)

        self._metric = error.upper()

        self._max_moment = max_moment

    def evaluate(self, samples, probabilities):
        """
        Evaluates gradient (for probability only)
        Just calls gradient_wrt_probabilities() for now
        """
        samples = self.check_bounds(samples.flatten(), self.get_param_bounds(joint_opt=self._joint_opt))
        # SROM defined by the current values of samples/probabilities for stats
        self.srom.set_params(samples, probabilities)

        if self._joint_opt:
            result = np.hstack(
                (self._gradient_wrt_samples(samples, probabilities), self._gradient_wrt_probabilities(samples)))
        else:
            result = self._gradient_wrt_probabilities(samples)

        return result

    def get_param_bounds(self, joint_opt):
        """
        Get the bounds on parameters for SROM optimization problem. If doing
        joint optimization, need bounds for both samples & probabilities. If
        not, just need trivial bounds on probabilities
        """

        if not joint_opt:
            bounds = [(0.0, 1.0)] * self.srom.size
        else:
            bounds = list(zip(self._target.mins, self._target.maxs)) * self.srom.size + [(0.0, 1.0)] * self.srom.size

        return bounds

    def check_bounds(self, samples, bounds):
        for i in range(self.srom.size):
            x = samples[i]
            if(x <= bounds[i][0]) or (x >= bounds[i][1]):
                samples[i] = np.clip(x, bounds[i][0] + 1e-2, bounds[i][1] - 1e-2)
        return samples.reshape(self.srom.size, self.srom.dim)

    def _gradient_wrt_samples(self, samples, probabilities):
        """
        Returns gradient vector w/ derivative of obj function w.r.t. SROM
        samples (m x d array)
        """

        srom_size = self.srom.size
        srom_dim = self.srom.dim

        # d_e1/d_x:
        if self._weights[0] > 0:
            cdf_grad = self._cdf_wrt_samples(samples, probabilities)
        else:
            cdf_grad = np.zeros((srom_size, srom_dim))

        # d_e2/d_x
        if self._weights[1] > 0:
            moment_grad = self._moment_wrt_samples(samples, probabilities)
        else:
            moment_grad = np.zeros((srom_size, srom_dim))

        # d_e3/d_x
        if self._weights[2] > 0:
            corr_grad = self._corr_wrt_samples(samples, probabilities)
        else:
            corr_grad = np.zeros((srom_size, srom_dim))

        grad = (-1 * self._weights[0] * cdf_grad +
                self._weights[1] * moment_grad +
                self._weights[2] * corr_grad)

        return grad.flatten()

    # def _cdf_integrand(self, x, x_srom, d, sig):
    #     srom_cdf = self.srom.compute_cdf(x, d, sig)
    #     target_cdf = self._target.compute_cdf(x)
    #     result = (srom_cdf - target_cdf) * np.exp(-1 / (2 * sig ** 2) * (x - x_srom) ** 2)
    #     return result

    def _cdf_wrt_samples(self, samples, probabilities):

        if samples.ndim > 1:
            (size, dim) = samples.shape
        else:
            (size, dim) = (samples.size, 1)

        grad = np.zeros((size, dim))

        # Compute relative diffs btwn srom/target CDFs.
        # Do a compute on the generated grid to get interpolants
        srom_cdfs = self.srom.compute_cdf(self._x_grid)
        target_cdfs = self._target.compute_cdf(self._x_grid)

        # Check for 0 cdf values to prevent divide by zero.
        i_nonzero = np.where(target_cdfs[:, 0] > 0)[0]
        srom_cdfs = srom_cdfs[i_nonzero, :]
        target_cdfs = target_cdfs[i_nonzero, :]
        diffs = (srom_cdfs - target_cdfs) / target_cdfs ** 2.0

        const = np.sqrt(2 * np.pi * self._scale ** 2)
        for j in range(dim):
            for srom_ind in range(size):
                x_srom = samples[srom_ind, j]
                grad[srom_ind, j] = np.sum(diffs[i_nonzero, j] * np.exp((-1 / (2 * self._scale ** 2)) *
                                                                        (self._x_grid[i_nonzero, j] - x_srom) ** 2))
                grad[srom_ind, j] *= (probabilities[srom_ind] / const)

        return grad

    def _moment_wrt_samples(self, samples, probabilities):

        if samples.ndim > 1:
            (size, dim) = samples.shape
        else:
            (size, dim) = (samples.size, 1)

        # Compute relative diffs between srom/target moments.
        srom_moments = self.srom.compute_moments(self._max_moment)

        # Reshape target moments to 2D if returned as 1D for scalar RV:
        target_moments = self._target.compute_moments(self._max_moment)
        if len(target_moments.shape) == 1:
            target_moments = target_moments.reshape((self._max_moment, 1))

        # Prevent divide by zero
        zero_indices = np.where(np.abs(target_moments) < 1e-12)[0]
        target_moments[zero_indices] = 1.0

        diffs = (srom_moments - target_moments) / target_moments ** 2.0

        # Compute gradient in obscure-looking but fast/vectorized way.
        samples_flat = samples.flatten()
        grad = np.zeros((size, dim))
        for q in range(self._max_moment):
            samples_q = np.multiply(np.tile((q + 1) * probabilities, dim), samples_flat ** q)
            diffs_tiled = np.tile(diffs[q, :], size)
            diffs_samples_q = np.multiply(samples_q, diffs_tiled)
            grad += diffs_samples_q.reshape(size, dim)

        return grad

    def _corr_wrt_samples(self, samples, probabilities):

        """
        Gradient of corr. error term with respect to samples (for srom_ind)
        """

        (size, dim) = samples.shape

        # Correlation irrelevant for 1D.
        if dim == 1:
            return np.zeros((size, dim))

        # Compute relative diffs between SROM/target correlation matrices.
        srom_corr = self.srom.compute_corr_mat()
        target_corr = self._target.compute_correlation_matrix()
        diffs = (srom_corr - target_corr) / target_corr ** 2.0

        grad = np.zeros((size, dim))

        def delta(idx_i, idx_j):
            return 0 if idx_i != idx_j else 1

        for srom_ind in range(size):
            sample_k = samples[srom_ind, :]
            prob_k = probabilities[srom_ind]

            for i in range(dim):
                grad_sum = 0.0
                for j in range(dim):
                    grad_sum += diffs[i, j] * prob_k * (delta(j, j) * sample_k[j] + delta(j, i) * sample_k[i])

                grad[srom_ind, i] = grad_sum

        return grad

    def _gradient_wrt_probabilities(self, samples):
        """
        Returns gradient vector w/ derivative of obj function w.r.t. SROM
        probabilities (m x 1 array)
        """

        srom_size = self.srom.size

        # d_e1/d_p:
        if self._weights[0] > 0:
            cdf_grad = self._cdf_wrt_prob(samples)
        else:
            cdf_grad = np.zeros(srom_size)

        # d_e2/d_p
        if self._weights[1] > 0:
            moment_grad = self._moment_wrt_prob(samples)
        else:
            moment_grad = np.zeros(srom_size)

        # d_e3/d_p
        if self._weights[2] > 0:
            corr_grad = self._corr_wrt_prob(samples)
        else:
            corr_grad = np.zeros(srom_size)

        grad = (self._weights[0] * cdf_grad +
                self._weights[1] * moment_grad +
                self._weights[2] * corr_grad)

        return grad

    def _cdf_wrt_prob(self, samples):
        """
        Gradient of CDF error term with respect to probability (for srom_ind)
        
        -Expression - the "erf" term of the gradient from the SROM paper 
        becomes an indicator function when smooth CDF is not used
        """

        (size, dim) = samples.shape

        # Compute relative diffs btwn srom/target CDFs.
        srom_cdfs = self.srom.compute_cdf(self._x_grid)
        target_cdfs = self._target.compute_cdf(self._x_grid)

        # Check for 0 cdf values to prevent divide by zero.
        i_nonzero = np.where(target_cdfs[:, 0] > 0)[0]
        srom_cdfs = srom_cdfs[i_nonzero, :]
        target_cdfs = target_cdfs[i_nonzero, :]
        diffs = (srom_cdfs - target_cdfs) / target_cdfs ** 2.0

        grad = np.zeros(size)

        for srom_ind in range(size):

            samples_k = samples[srom_ind, :]
            grad_i = 0

            for i in range(dim):
                grid_i = self._x_grid[i_nonzero, i]

                # Implement indicator function in vectorized way:
                indices = grid_i >= samples_k[i]
                grad_i += np.sum(diffs[indices, i])

            grad[srom_ind] = grad_i

        return grad

    def _moment_wrt_prob(self, samples):
        """
        Gradient of moment error term with respect to probability (for srom_ind)
        """

        (size, dim) = samples.shape

        # Compute relative diffs between srom/target moments.
        srom_moments = self.srom.compute_moments(self._max_moment)

        # Reshape target moments to 2D if returned as 1D for scalar RV:
        target_moments = self._target.compute_moments(self._max_moment)
        if len(target_moments.shape) == 1:
            target_moments = target_moments.reshape((self._max_moment, 1))

        # Prevent divide by zero
        zero_indices = np.where(np.abs(target_moments) < 1e-12)[0]
        target_moments[zero_indices] = 1.0

        diffs = (srom_moments - target_moments) / target_moments ** 2.0

        # Compute gradient in obscure-looking but fast/vectorized way.
        samples_flat = samples.flatten()
        grad = np.zeros(size)
        for q in range(self._max_moment):
            samples_q = samples_flat ** (q + 1)
            diffs_tiled = np.tile(diffs[q, :], size)
            diffs_samples_q = np.multiply(samples_q, diffs_tiled)
            grad += np.sum(diffs_samples_q.reshape(size, dim), axis=1)

        return grad

    def _corr_wrt_prob(self, samples):
        """
        Gradient of corr. error term with respect to probability (for srom_ind)
        """

        (size, dim) = samples.shape

        # Correlation irrelevant for 1D.
        if dim == 1:
            return np.zeros(size)

        # Compute relative diffs between SROM/target correlation matrices.
        srom_corr = self.srom.compute_corr_mat()
        target_corr = self._target.compute_correlation_matrix()
        diffs = (srom_corr - target_corr) / target_corr ** 2.0

        grad = np.zeros(size)

        for srom_ind in range(size):
            sample_k = samples[srom_ind, :]
            grad_sum = 0.0

            for i in range(dim):
                for j in range(dim):
                    grad_sum += diffs[i, j] * sample_k[i] * sample_k[j]

            grad[srom_ind] = grad_sum

        return grad

    def _generate_cdf_grids(self, cdf_grid_pts):
        """
        Generate numerical grids for evaluating the CDF errors based on the 
        range of the target random vector. Create x_grid member variable with
        cdf_grid_pts along each dimension of the random vector.
        """

        self._x_grid = np.zeros((cdf_grid_pts, self._target.dim))

        for i in range(self._target.dim):
            grid = np.linspace(self._target.mins[i],
                               self._target.maxs[i],
                               cdf_grid_pts)
            self._x_grid[:, i] = grid

    def __check_init_parameters(self, obj_weights, error, scale):

        if obj_weights is not None:
            if len(obj_weights) != 3:
                raise ValueError("obj_weights must have length 3!")
            self._weights = obj_weights
        else:
            self._weights = np.ones((3,))

        if error.upper() not in ["MEAN", "MAX", "SSE"]:
            raise ValueError("error must be either 'mean','max', or 'sse'")

        if scale is not None:
            if isinstance(scale, int):
                scale = float(scale)
            if not isinstance(scale, float):
                raise TypeError("Smooth CDF scale must be numeric.")
