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

"""
Class for implementing a translation random vector for non-gaussian random
vectors whose components are governed by analytic probability distributions.
"""

import copy
import numpy as np

from scipy.stats import multivariate_normal, norm
from scipy import integrate, interpolate

from SROMPy.target.RandomVector import RandomVector


class AnalyticRandomVector(RandomVector):
    """
    Class for implementing a translation random vector for non-gaussian random
    vectors whose components are governed by analytic probability distributions
    and have known correlation.

    :param random_variables: list of SROMPy target random variable objects 
                             defining each component of the random vector.
    :type random_variables: list of SROMPy random variable objects
    :param correlation_matrix: specifies correlation between vector components.
    :type correlation_matrix: np array, size: dim x dim

    random_variables list must have length equal to the random vector dimension.
    Each SROMPy random variable object in the list must be properly
    initialized and have compute_moments and compute_CDF functions implemented.
    """

    def __init__(self, random_variables, correlation_matrix):
        """
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
        """

        # TODO - error checking to make sure random variables are properly
        # initialized / constructed / necessary functions / member variables
        # like _min / _max

        # Error checking on correlation matrix:
        self.verify_correlation_matrix(correlation_matrix)
        self._corr = copy.deepcopy(correlation_matrix)

        # Size of correlation matrix must match # random variable components:
        if self._corr.shape[0] != len(random_variables):
            raise ValueError("Dimension mismatch btwn corr mat & random vars")

        # Parent class (RandomVector) constructor, sets self._dim
        super(AnalyticRandomVector, self).__init__(len(random_variables))

        self._gaussian_corr = None
        self._unscaled_correlation = None

        # Get min/max values for each component.
        self._components = copy.deepcopy(random_variables)

        self.mins = np.zeros(self._dim)
        self.maxs = np.zeros(self._dim)

        for i in range(self._dim):
            self.mins[i] = self._components[i].mins[0]
            self.maxs[i] = self._components[i].maxs[0]

        # Generate Gaussian correlation matrix for sampling translation RV:
        self.generate_gaussian_correlation()

        # Generate unscaled correlation that is matched by SROM during opt.
        self.generate_unscaled_correlation()

    @staticmethod
    def verify_correlation_matrix(corr_matrix):
        """
        Do error checking on the provided correlation matrix, e.g., is it
        square? is it symmetric?
        """

        corr_matrix = np.array(corr_matrix)  # Make sure it's an numpy array.

        if len(corr_matrix.shape) == 1:
            raise ValueError("Correlation matrix must be a 2D array!")

        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square!")

        # Slick check for symmetry:
        if not np.allclose(corr_matrix, corr_matrix.T, 1e-6):
            raise ValueError("Correlation matrix must be symmetric!")

        # Make sure all entries are positive:
        if np.any(corr_matrix < 0):
            raise ValueError("Correlation matrix entries must be positive!")

    def compute_moments(self, max_):
        """
        Calculate random vector moments up to order max_moment based
        on samples. Moments from 1,...,max_order
        """

        # Get moments up to max_ for each component of the vector.
        moments = np.zeros((max_, self._dim))
        for i in range(self._dim):
            moments[:, i] = self._components[i].compute_moments(max_).flatten()

        return moments

    def compute_cdf(self, x_grid):
        """
        Evaluates the precomputed/stored CDFs at the specified x_grid values
        and returns. x_grid can be a 1D array in which case the CDFs for each
        dimension are evaluated at the same points, or it can be a
        (num_grid_pts x dim) array, specifying different points for each
        dimension - each dimension can have a different range of values but
        must have the same # of grid pts across it. Returns a (num_grid_pts x
        dim) array of corresponding CDF values at the grid points
        """

        # NOTE - should deep copy x_grid since were modifying?
        # 1D random variable case
        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_pts, dim) = x_grid.shape

        # If only one grid was provided for multiple dims, repeat to generalize.
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        cdf_values = np.zeros((num_pts, self._dim))

        # Evaluate CDF interpolants on grid.
        for d, grid in enumerate(x_grid.T):

            # Make sure grid values lie within max/min along each dimension.
            grid[np.where(grid < self.mins[d])] = self.mins[d]
            grid[np.where(grid > self.maxs[d])] = self.maxs[d]

            cdf_values[:, d] = self._components[d].compute_cdf(grid)

        return cdf_values

    def compute_correlation_matrix(self):
        """
        Returns the correlation matrix
        """
        return self._unscaled_correlation

    def draw_random_sample(self, sample_size):
        """
        Implements the translation model to generate general random vectors with
        non-gaussian components. Nonlinear transformation of a std gaussian
        vector according to method in S.R. Arwade 2005 paper.

        random component sample: theta = inv_cdf(std_normal_cdf(normal_vec))
                                 \Theta = F^{-1}(\Phi(G))

        """

        samples = np.zeros((sample_size, self._dim))
        cholesky = np.linalg.cholesky(self._gaussian_corr)

        # Is there a way to optimize this sampling loop?
        for i in range(sample_size):

            # Draw standard std normal random vector with given correlation.
            normal_vector = cholesky*norm.rvs(size=self._dim)

            # Evaluate std normal CDF at the random vector.
            norm_cdf = norm.cdf(normal_vector)

            # Transform by inverse CDF of random vector's components.
            for j in range(self._dim):
                samples[i][j] = \
                    self._components[j].compute_inv_cdf(norm_cdf[j])[0]

        return samples

    def integrand_helper(self, u, v, k, j, rho_kj):
        """
        Helper function for numerical integration in the
        generate_gaussian_correlation() function. Implements the integrand of
        equation 6 of J.M. Emery 2015 paper that needs to be integrated w/
        scipy
        Passing in values of the k^th and j^th component of the random variable
        - u and v - and the specified correlation between them rho_kj.
        """

        normal_pdf_kj = multivariate_normal.pdf([u, v],
                                                cov=[[1, rho_kj], [rho_kj, 1]])

        # f_k(x) = InvCDF_k ( Gaussian_CDF( x ) ).
        f_k = self._components[k].compute_inv_cdf(norm.cdf(u))
        f_j = self._components[j].compute_inv_cdf(norm.cdf(v))

        integrand = f_k*f_j*normal_pdf_kj

        return integrand

    def get_correlation_entry(self, k, j, rho_kj):
        """
        Get the correlation between this random vector's k & j components from
        the correlation btwn the Gaussian random vector's k & j components.
        Helper function for generate_gaussian_correlation
        Need to integrate product of k/j component's inv cdf & a standard
        2D normal pdf with correlation rho_kj. This is equation 6 in J.M. Emery
        et al 2015.
        """

        # Integrate using scipy.
        k_lims = [-4, 4]
        j_lims = [-4, 4]

        # Get product of moments & std deviations for equation 6.
        mu_k_mu_j = (self._components[k].compute_moments(1)[0] *
                     self._components[j].compute_moments(1)[0])
        std_k_std_j = (self._components[k].get_variance() *
                       self._components[j].get_variance())**0.5

        # Try adjusting tolerance to speed this up:
        # 1.49e-8 is default for both.
        opts = {'epsabs': 1.e-5, 'epsrel': 1e-5}
        e_integral = integrate.nquad(self.integrand_helper, [k_lims, j_lims],
                                     args=(k, j, rho_kj), opts=opts)

        eta_kj = (e_integral - mu_k_mu_j)/std_k_std_j

        return eta_kj[0]

    def generate_gaussian_correlation(self):
        """
        Generates the Gaussian correlation matrix that will achieve the
        covariance matrix specified for this random vector when using a
        translation random vector sampling approach. See J.M. Emery 2015 paper
        pages 922,923 on this procedure.
        Helper function - no inputs, operates on self._correlation correlation
        matrix and generates self._gaussian_corr
        """

        self._gaussian_corr = np.ones(self._corr.shape)

        # Want to build interpolant from eta correlation values to rho
        # correlation values.
        num_points = 12

        # -1 made matrix singular
        rho_kj_grid = np.linspace(-0.99, 0.99, num_points)
        eta_jk_grid = np.zeros(num_points)

        for k in range(self._dim):
            for j in range(k+1, self._dim):
                print("Determining correlation entry ", k, " ", j)

                # Compute grid of eta/rho pts:
                for i, rho_kj in enumerate(rho_kj_grid):
                    eta_jk_grid[i] = self.get_correlation_entry(k, j, rho_kj)

                # Build interpolant to find rho value for specified eta.
                rho_interp = interpolate.interp1d(eta_jk_grid, rho_kj_grid)

                # Use symmetry to save time:
                self._gaussian_corr[k][j] = rho_interp(self._corr[k][j])
                self._gaussian_corr[j][k] = rho_interp(self._corr[k][j])

    def generate_unscaled_correlation(self):
        """
        Generates the unscaled correlation matrix that is matched by the SROM
        during optimization. No inputs / outputs. INternally produces
        self._unscaled_correlation from self._correlation.

        >> C_ij = E[ X_i X_j]

        """

        self._unscaled_correlation = copy.deepcopy(self._corr)

        for i in range(self._dim):
            for j in range(self._dim):
                mu_i_mu_j = (self._components[i].compute_moments(1)[0] *
                             self._components[j].compute_moments(1)[0])
                std_i_std_j = (self._components[i].get_variance() *
                               self._components[j].get_variance())**0.5
                self._unscaled_correlation[i][j] *= std_i_std_j
                self._unscaled_correlation[i][j] += mu_i_mu_j
