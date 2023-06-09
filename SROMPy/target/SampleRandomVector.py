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
Class for defining a sample-based random vector with empirical estimators
"""

import numpy as np
from scipy import interpolate
from scipy.stats.qmc import Halton, Sobol

from SROMPy.target import RandomVector


class SampleRandomVector(RandomVector):
    """
    Sample-based random vector. Defines a target random vector to match with
    an SROM based on a set of realizations of that random vector. Implements
    basic statistics to use in SROM optimization and comparisons.

    :param samples: set of realizations/samples of the random vector
    :type samples: np array, size: (# samples x dim)
    :param max_moment: max. order moment to precompute and store
    :type max_moment: int
    """

    def __init__(self, samples, max_moment=10):
        """
        Initialize SampleRandomVector with an array of samples of the random
        vector. Must be an array of size (# samples x dim). Statistics of the
        SampleRandomVector are precomputed during initialization - max_moment
        is the maximum moment order to compute & store for later use. If higher
        moments are anticipated, this can be increased (or visa versa)
        """

        # Check for 1D case (random variable).
        if len(samples.shape) == 1:
            samples = samples.reshape((len(samples), 1))

        (num_samples, dim) = samples.shape

        if dim > num_samples:
            msg = "Dimension is greater than # samples! Check samples array"
            raise ValueError(msg)

        self._CDFs = []
        self.mins = []
        self.maxs = []

        # Parent class (RandomVector) constructor, sets self._dim.
        super(SampleRandomVector, self).__init__(dim)

        self._num_samples = num_samples
        self._samples = samples
        self._max_moment = max_moment
        self._moments = None
        self._correlation = None

        # Precompute & store statistics so they can be returned quickly later.
        self.generate_statistics(max_moment)

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def samples(self):
        return self._samples

    def compute_moments(self, max_order):
        """
        Return precomputed moments up to specified order.

        :param max_order: Maximum order of moments to return
        :type max_order: int

        Returns (max_order x dim) size Numpy array with SROM moments for
        each dimension.

        """

        # TODO - calculate moments above max_moment on the fly &
        # append to stored
        if max_order <= self._max_moment:
            moments = self._moments[:max_order, :]
        else:
            raise NotImplementedError("Moment above max_moment not handled yet")

        return moments

    def compute_cdf(self, x_grid):
        """
        Evaluates the precomputed/stored CDFs at the specified x_grid values
        and returns. 

        :param x_grid: Grid of points to compute CDF values on. If 1d array is
            provided, the same points are used to evaluate CDF in each
            dimension. If 2d array is provided, calculates CDF values on
            different points, but must have same # points for each dimension.
            Size is (# grid pts) x (dim) or (# grid pts) x (1).
        :type x_grid: Numpy array.

        Returns: Numpy array of CDF values at x_grid points. Size is (# grid
        pts) x (dim).
        """

        # 1D random variable case.
        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_points, dim) = x_grid.shape

        # If only one grid was provided for multiple dims, repeat to generalize.
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        cdf_values = np.zeros((num_points, self._dim))

        # Evaluate CDF interpolants on grid.
        for d, grid in enumerate(x_grid.T):

            # Make sure grid values lie within max/min along each dimension.
            grid[np.where(grid < self.mins[d])] = self.mins[d]
            grid[np.where(grid > self.maxs[d])] = self.maxs[d]

            cdf_values[:, d] = self._CDFs[d](grid)

        return cdf_values

    def compute_correlation_matrix(self):
        """
        Returns precomputed correlation matrix.
        """
        return self._correlation

    def draw_random_sample(self, sample_size, qmc_engine=None):
        """
        Randomly draws a sample of this random vector.

        :param sample_size: number of samples to return
        :type sample_size: int

        sample_size must be smaller than total # of samples. For sample-based
        random vector, we return a randomly selected # of samples
        """

        if sample_size > self._num_samples:
            raise ValueError("Sample size can't be more than total # samples")

        if qmc_engine is not None:
            if qmc_engine == 'Halton':
                sampler = Halton(d=self._dim)
                random_indices = sampler.integers(l_bounds=0, u_bounds=self._num_samples, n=sample_size)
            elif qmc_engine == 'Sobol':
                sampler = Sobol(d=self._dim)
                random_indices = sampler.integers(l_bounds=0, u_bounds=self._num_samples, n=sample_size)
            else:
                raise ValueError("Invalid QMC engine provided.")
        else:
            # Generate random indices for samples array.
            all_indices = np.arange(self._num_samples)
            random_indices = np.random.choice(all_indices, sample_size, replace=False)

        sample = self._samples[random_indices, :]

        return sample

    # --------------Helper initialization methods---------------------------

    def get_plot_cdfs(self):
        """
        Get CDF values for plotting (without using interpolant) - returns
        tuple with x_grid & CDF values arrays
        """

        x_grid = np.zeros((self._num_samples, self._dim))
        cdf_values = np.zeros((self._num_samples, self._dim))

        for i, samples_i in enumerate(self._samples.T):

            # Generate empirical CDF:
            sorted_i = np.sort(samples_i)
            cdf_vals_i = np.arange(1, len(sorted_i) + 1)/float(len(sorted_i))
            x_grid[:, i] = sorted_i
            cdf_values[:, i] = cdf_vals_i

        return x_grid, cdf_values

    def generate_statistics(self, max_moment):
        """
        Precompute & store moments, CDFs, correlation matrix of the samples
        so that they can be returned quickly later
        """

        self.generate_moments(max_moment)
        self.generate_cdfs()
        self.generate_correlation()

    def generate_moments(self, max_moment):
        """
        Calculate & store random vector moments up to order max_moment based
        on samples. Moments from 1,...,max_order
        """

        self._moments = np.zeros((max_moment, self._dim))

        factor = (1./float(self._num_samples))
        for q in range(0, max_moment):

            moment_q = np.zeros((1, self._dim))
            for sample in self._samples:
                moment_q += factor*np.power(sample, q+1)

            self._moments[q, :] = moment_q

    def generate_cdfs(self):
        """
        Calculate & store marginal CDFs for each dimension of the random vector.
        Stores a linear interpolator of the CDF for each dim
        Uses trick from :   http://stackoverflow.com/questions/3209362/
                            %20how-to-plot-empirical-cdf-in-matplotlib-in-python
        to calculate CDF from samples
        """

        self._CDFs = []

        # Need to store max/min samples in each dimension to prevent out of
        # bounds values in the interpolators later.
        self.mins = []
        self.maxs = []

        # Get all samples of the i^th dimension at a time to generate CDF
        # NOTE - does iterating over / sorting happen in place? Need deep copy?
        for samples_i in self._samples.T:

            # Generate/store interpolant for empirical CDF:
            sorted_i = np.sort(samples_i)
            cdf_values = np.arange(1, len(sorted_i) + 1)/float(len(sorted_i))
            cdf_func = interpolate.interp1d(sorted_i, cdf_values)
            self._CDFs.append(cdf_func)

            self.mins.append(sorted_i[0])
            self.maxs.append(sorted_i[-1])

    def generate_correlation(self):
        """
        Calculates and stores sample-based correlation matrix for random vector
        """

        # TODO - find faster numpy/scipy function
        self._correlation = np.zeros((self._dim, self._dim))

        factor = (1./float(self._num_samples))
        for sample in self._samples:
            self._correlation = self._correlation +\
                                factor * np.outer(sample, sample)
