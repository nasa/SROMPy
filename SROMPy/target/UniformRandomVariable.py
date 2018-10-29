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
Class for defining a uniform random variable
"""

import numpy as np
from scipy.stats import uniform as scipy_uniform

from SROMPy.target.RandomVariable import RandomVariable


class UniformRandomVariable(RandomVariable):
    """
    Class for defining a uniform random variable
    """

    def __init__(self, min_val=0., max_val=0., max_moment=10):
        """
        Initialize the uniform (gaussian) random variable with provided
        minimum/maximum values. Implementation wraps scipy.stats.uniform to get
        statistics/samples. Caches moments up to max_moment for speedup.
        """

        if min_val >= max_val:
            raise ValueError("Minimum value must be less than maximum value")

        self._minimum_value = min_val
        self._range_size = max_val - min_val
        self._moments = None

        # Set dimension (scalar), min/max to equal mean +/- 4stds.
        self.dim = 1
        self.mins = [min_val]
        self.maxs = [max_val]

        # Cache moments.
        self.generate_moments(max_moment)
        self._max_moment = max_moment

    def get_variance(self):
        """
        Returns variance of uniform random variable
        """
        return self._std**2.0

    def compute_moments(self, max_order):
        """
        Returns moments up to order 'max_order' in numpy array.
        """

        # TODO - calculate moments above max_moment on the fly &
        # append to stored
        if max_order <= self._max_moment:
            moments = self._moments[:max_order]
        else:
            raise NotImplementedError("Moment above max_moment not handled yet")

        return moments

    def compute_cdf(self, x_grid):
        """
        Returns numpy array of uniform CDF values at the points contained
        in x_grid.
        """

        return scipy_uniform.cdf(x_grid, self._minimum_value, self._range_size)

    def compute_inv_cdf(self, x_grid):
        """
        Returns np array of inverse uniform CDF values at pts in x_grid
        """
        return scipy_uniform.ppf(x_grid, self._minimum_value, self._range_size)

    def compute_pdf(self, x_grid):
        """
        Returns numpy array of uniform pdf values at the points contained
        in x_grid
        """
        return scipy_uniform.pdf(x_grid, self._minimum_value, self._range_size)

    def draw_random_sample(self, sample_size):
        """
        Draws random samples from the uniform random variable. Returns numpy
        array of length 'sample_size' containing these samples
        """

        # Use scipy uniform rv to return shifted/scaled samples automatically.
        return scipy_uniform.rvs(self._minimum_value, self._range_size,
                                 sample_size)

    def generate_moments(self, max_moment):
        """
        Calculate & store moments to retrieve more efficiently later
        """

        self._moments = np.zeros((max_moment, 1))

        # Rely on scipy.stats to return non-central moment.
        for i in range(max_moment):
            self._moments[i] = scipy_uniform.moment(i + 1, self._minimum_value,
                                                    self._range_size)
