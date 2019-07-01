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
Class for implementing a beta random variable
"""

import numpy as np
from scipy.stats import beta as scipy_beta

from SROMPy.target.RandomVariable import RandomVariable


class BetaRandomVariable(RandomVariable):
    """
    Class for implementing a beta random variable
    """

    def __init__(self, alpha, beta, shift=0, scale=1, max_moment=10):
        """
        Initialize the beta random variable with the standard alpha and beta
        shape parameters (follows convention for a & b in numpy.random.beta).
        Optionally specify shift & scale parameters to translate and scale the
        random variable, e.g.:
            new_beta = shift + scale * standard_beta.

        If one wants to specify a beta random variable to match a given
        support (min, max), mean, and variance, use the static method
        get_beta_shape_params() to convert to inputs for this constructor.

        Implementation wraps scipy.stats.beta to get statistics/samples.
        """

        if alpha < 0:
            raise ValueError("Alpha shape param must be non-negative")
        if beta < 0:
            raise ValueError("Beta shape param must be non-negative")
        if scale <= 0:
            raise ValueError("Scale param must be positive")

        self._alpha = alpha
        self._beta = beta
        self._shift = shift
        self._scale = scale
        self._moments = None

        # Set dimension (scalar), min/max.
        self._dim = 1
        self.mins = [shift]
        self.maxs = [shift + scale]

        # Cache moments.
        self.generate_moments(max_moment)
        self._max_moment = max_moment

    @staticmethod
    def get_beta_shape_params(min_value, max_value, mean, variance):
        """
        Returns the beta shape parameters (alpha, beta) and the shift/scale
        parameters that produce a beta random variable with the specified
        minimum value, maximum value, mean, and variance. Can be called prior
        to initialization of this class if only this info is known about the
        random variable being modeled.
        Returns a list of length 4 ordered [alpha, beta, shift, scale]
        """

        # Cast to make sure we have floats for calculations.
        min_value = float(min_value)
        max_value = float(max_value)
        mean_val = float(mean)
        variance = float(variance)

        # Scale mean/variance to lie in [0,1] for standard beta distribution.
        mean_std = (mean_val - min_value) / (max_value - min_value)
        var_std = (1. / (max_value - min_value)) ** 2.0 * variance

        # Get shape params based on scaled mean/variance:
        alpha = mean_std*(mean_std*(1. - mean_std) / var_std - 1.)
        beta = (mean_std*(1 - mean_std)/var_std - 1) - alpha
        shift = min_value
        scale = max_value - min_value

        return [alpha, beta, shift, scale]

    def get_variance(self):
        """
        Returns variance of beta random variable
        """
        a = self._alpha
        b = self._beta
        var = (a*b)/(((a+b)**2) * (a + b + 1))*self._scale**2
        return var

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
        Returns numpy array of beta CDF values at the points contained in x_grid
        """

        return scipy_beta.cdf(x_grid, self._alpha, self._beta, self._shift,
                              self._scale)

    def compute_inv_cdf(self, x_grid):
        """
        Returns np array of inverse beta CDF values at pts in x_grid
        """
        return scipy_beta.ppf(x_grid, self._alpha, self._beta, self._shift,
                              self._scale)

    def compute_pdf(self, x_grid):
        """
        Returns numpy array of beta pdf values at the points contained in x_grid
        """
        return scipy_beta.pdf(x_grid, self._alpha, self._beta, self._shift,
                              self._scale)

    def draw_random_sample(self, sample_size):
        """
        Draws random samples from the beta random variable. Returns numpy
        array of length 'sample_size' containing these samples
        """

        # Use scipy beta rv to return shifted/scaled samples automatically.
        return scipy_beta.rvs(self._alpha, self._beta, self._shift, self._scale,
                              sample_size)

    def generate_moments(self, max_moment):
        """
        Calculate & store moments to retrieve more efficiently later
        """

        self._moments = np.zeros((max_moment, 1))

        # Rely on scipy.stats to return non-central moment.
        for i in range(max_moment):
            self._moments[i] = scipy_beta.moment(i + 1, self._alpha, self._beta,
                                                 self._shift, self._scale)
