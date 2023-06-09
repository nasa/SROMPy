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
Define stochastic reduced order model (SROM) class
"""

import copy
import os
import numpy as np
from scipy.special import erf

from SROMPy.optimize import Optimizer
from SROMPy.target.RandomEntity import RandomEntity


class SROM(object):
    """
    This is the primary SROMPy class for defining and utilizing a
    stochastic reduced order model (SROM). Main capability is optimizing
    for the defining SROM parameters to model a given target random quantity.
    Other functions provided to calculate SROM statistics, set/get defining
    parameters directly, and store/load SROM to/from file.

    :param size: SROM size
    :type size: int
    :param dim: dimension of random quantity being modeled
    :type dim: int
    """

    def __init__(self, size, dim):
        """
        Initialize SROM w/ specified size for random vector of dimension dim
        m = SROM size, d = dimension;
        """

        if size <= 0:
            raise(ValueError("SROM size must be greater than zero."))

        if dim <= 0:
            raise(ValueError("SROM dimension must be greater than 0."))

        self._size = int(size)
        self._dim = int(dim)

        self.samples = None
        self.probabilities = None
        self._scale = None  # smooth CDF approximation

    @property
    def size(self):
        return self._size

    @property
    def dim(self):
        return self._dim

    def set_params(self, samples, probabilities):
        """
        Set defining SROM parameters - samples & corresponding probabilities.

        :param samples: Array of SROM samples
        :type samples: 2d Numpy array, size - (SROM size) x (dim)
        :param probabilities: Array of SROM probabilities
        :type probabilities: 1d Numpy array, size - (SROM size) x 1

        The sample/probability arrays have the following convention (srom sample
        index as rows, components of sample as columns):

        Samples:

        | [[ x_1^(1),   x_2^(1), ..., x_d^(1)],
        | [x_1^(2), x_2^(2), ..., x_d^(2)],
        | ...     ...   ...    ....
        | [x_1^(m), x_2^(m),  ...  x_d^(m)]]
 
        Probabilities:

        | [p^(1), p^(2), ..., p^(m)]^T

        """

        # Handle 1 dimension case, adjust shape:
        if len(samples.shape) == 1:
            samples.shape = (len(samples), 1)

        # Verify dimensions of samples/probabilities.
        (size, dim) = samples.shape

        if size != self._size and dim != self._dim:
            msg = "SROM samples have wrong dimension, must be (srom_size x dim)"
            raise ValueError(msg)

        if len(probabilities) != self._size:
            raise ValueError("SROM probabilities must have dim. equal to srom "
                             "size")

        self.samples = copy.deepcopy(samples)
        self.probabilities = \
            copy.deepcopy(probabilities.reshape((self._size, 1)))

    def get_params(self):
        """
        Returns: tuple of SROM sample & probability arrays. Samples array
        has size (SROM size x dim) and probability array has length (SROM size)

        The sample/probability arrays have the following convention (srom sample
        index as rows, components of sample as columns):

        Samples:

        | [[ x_1^(1),   x_2^(1), ..., x_d^(1)],
        | [x_1^(2), x_2^(2), ..., x_d^(2)],
        | ...     ...   ...    ....
        | [x_1^(m), x_2^(m),  ...  x_d^(m)]]

        Probabilities:

        | [p^(1), p^(2), ..., p^(m)]^T

        """
        return self.samples, self.probabilities

    def compute_moments(self, max_order):
        """
        Calculates and returns SROM moments.

        :param max_order: Maximum order of moments to return
        :type max_order: int

        Returns (max_order x dim) size Numpy array with SROM moments for
        each dimension.
        """

        # Make sure SROM has been properly initialized.
        if self.samples is None or self.probabilities is None:
            raise ValueError("Must initialize SROM before computing moments")

        max_order = int(max_order)
        moments = np.zeros((max_order, self._dim))

        for q in range(max_order):

            # moment_q = sum_{k=1}^m p(k) * x(k)^q.
            moment_q = np.zeros((1, self._dim))
            for k, sample in enumerate(self.samples):
                moment_q = moment_q + self.probabilities[k] * pow(sample, q + 1)

            moments[q, :] = moment_q

        return moments

    def compute_cdf(self, x_grid):
        """
        Computes the SROM marginal CDF values in each dimension.

        :param x_grid: Grid of points to compute CDF values on. If 1d array is
            provided, the same points are used to evaluate CDF in each
            dimension. If 2d array is provided, calculates CDF values on
            different points, but must have same # points for each dimension.
            Size is (# grid pts) x (dim) or (# grid pts) x (1).
        :type x_grid: Numpy array.

        :param sigma: scaling parameter for the smooth CDF approximation. If
            None, uses the empirical CDF computed from an indicator function.
            Otherwise, uses the error function approximation.
        :type sigma: float

        Returns: Numpy array of CDF values at x_grid points. Size is (# grid
        pts) x (dim).

        Note:
            * Increasing the number of grid points can significantly slow
              down the SROM optimization problem.
            * Providing a 2d array for x_grid can specify a different range
              of values for each dimension, but must use the same number of pts.
        """

        # Make sure SROM has been properly initialized
        if self.samples is None or self.probabilities is None:
            raise ValueError("Must initialize SROM before computing CDF")

        if len(x_grid.shape) == 1:
            x_grid = x_grid.reshape((len(x_grid), 1))
        (num_pts, dim) = x_grid.shape

        # If only one grid was provided for multiple dims, repeat to generalize.
        if (dim == 1) and (self._dim > 1):
            x_grid = np.repeat(x_grid, self._dim, axis=1)

        if self._scale is not None:
            cdf_values = self._compute_cdf_smooth(num_pts, x_grid, self._scale)
        else:
            cdf_values = self._compute_cdf_empirical(num_pts, x_grid)

        return cdf_values

    def _compute_cdf_empirical(self, num_pts, x_grid):

        cdf_values = np.zeros((num_pts, self._dim))

        # Vectorized indicator implementation for CDF.
        # CDF(x) = sum_{k=1}^m  1( sample^(k) < x) prob^(k).
        for i, grid in enumerate(x_grid.T):
            for k, sample in enumerate(self.samples):

                indices = grid >= sample[i]
                cdf_values[indices, i] += self.probabilities[k]

        return cdf_values

    def _compute_cdf_erf(self, x, d, sigma):

        cdf_value = 0.0
        for k in range(self._size):
            cdf_value += 0.5 * self.probabilities[k]*(1.0 + erf((x - self.samples[k, d]) / (np.sqrt(2) * sigma)))

        return cdf_value

    def _compute_cdf_smooth(self, num_pts, x_grid, sigma):

        cdf_values = np.zeros((num_pts, self._dim))

        for i, grid in enumerate(x_grid.T):
            cdf_values[:, i] = self._compute_cdf_erf(grid, i, sigma=sigma)

        return cdf_values

    def compute_corr_mat(self):
        """
        Returns the SROM correlation matrix as (dim x dim) numpy array

        srom_corr = sum_{k=1}^m [ x^(k) * (x^(k))^T ] * p^(k)
        """

        # Make sure SROM has been properly initialized
        if self.samples is None or self.probabilities is None:
            raise ValueError("Must initialize SROM before computing moments")

        corr = np.zeros((self._dim, self._dim))

        for k, sample in enumerate(self.samples):
            corr = corr + np.outer(sample, sample) * self.probabilities[k]

        return corr

    def optimize(self, target_random_variable,
                 weights=None,
                 num_test_samples=50,
                 error='SSE',
                 max_moment=5,
                 cdf_grid_pts=100,
                 tolerance=None,
                 options=None,
                 method=None,
                 joint_opt=False,
                 opt_output_interval=10,
                 verbose=True,
                 scale=None):
        """
        Optimize for the SROM samples & probabilities to best match the
        target random vector statistics. The main functionality provided
        by the SROM class. Solves SROM the optimization problem and sets
        the samples and probabilities for the SROM object to the optimized
        values.

        :param target_random_variable: the target random quantity
            (variable/vector) being modeled by the SROM.
        :type target_random_variable: SROMPy target object
            (AnalyticRandomVector, SampleRandomVector, or random variable class)
        :param weights: relative weights specifying importance of matching
            CDFs, moments, and correlation of the target during optimization.
            Default is equal weights [1,1,1].
        :type weights: 1d Numpy array (length = 3)
        :param num_test_samples: Number of sample sets (iterations) to run
            optimization.
        :type num_test_samples: int
        :param error: Type of error metric to use in objective ("SSE", "MAX",
            "MEAN").
        :type error: string
        :param max_moment: Max. number of target moments to consider matching
        :type max_moment: int
        :param cdf_grid_pts: Number of points to evaluate CDF error on
        :type cdf_grid_pts: int
        :param tolerance: tolerance for scipy optimization algorithm (TODO)
        :type tolerance: float
        :param options: scipy optimization algorithm options (TODO)
        :type options: dict
        :param method: method used for scipy optimization  (TODO)
        :type method: string
        :param joint_opt: Flag to optimize jointly for samples & probabilities.
        :type joint_opt: bool
        :param opt_output_interval: # iterations to skip before printing output
        :type opt_output_interval: int
        :param verbose: flag indicating to print optimization status to stdout
        :type verbose: bool
        :param scale: the scale for the smooth CDF approximation
        :type scale: float

        Returns: None. Sets samples/probabilities member variables.

        Assumes the targetRV object has been properly initialized beforehand.
        The optimization for SROM samples & probabilities is currently
        performed sequentially - a random set of samples are first drawn and the
        probabilities are then optimization for those samples. The input
        "num_test_samples" is the number of random sample sets this is
        performed for before terminating. The random sample set and optimal
        probabilities found that produce the lowest objective function value
        are used as the optimal parameters. The joint_opt input flag can 
        specify to do the optimization over samples and probabilities 
        simultaneously.
        """

        if not isinstance(target_random_variable, RandomEntity):
            raise TypeError("target_random_variable must inherit from "
                            "RandomEntity.")

        self._scale = scale
        # Use optimizer to form SROM objective func & gradient and minimize:
        opt = Optimizer(target_random_variable,
                        self,
                        weights,
                        error,
                        max_moment,
                        cdf_grid_pts,
                        joint_opt=joint_opt,
                        scale=scale)

        (samples, probabilities) = opt.get_optimal_params(num_test_samples,
                                                          tolerance,
                                                          options,
                                                          method,
                                                          joint_opt,
                                                          opt_output_interval,
                                                          verbose)

        self.set_params(samples, probabilities)

    def save_params(self, outfile="srom_params.txt", delimiter=' '):
        """
        Write the SROM parameters to file.
 
        :param outfile: output file name
        :type outfile: string
        :param delimiter: delimiter used in output file (default - whitespace)
        :type delimiter: string

        Returns: None. Produces output file.

        Writes output file with the following format (samples in each row with
        prob after):

        | x_1^(1),   x_2^(1), ..., x_d^(1),  p^(1)
        | x_1^(2), x_2^(2), ..., x_d^(2),  p^(2)
        | ...     ...   ...    ....   ...
        | x_1^(m), x_2^(m),  ...     x_d^(m),  p^(m)

        """

        # Make sure SROM has been properly initialized
        if self.samples is None or self.probabilities is None:
            raise ValueError("Must initialize SROM before saving to disk")

        srom_params = np.hstack((self.samples, self.probabilities))
        np.savetxt(outfile, srom_params, delimiter=delimiter)

    def load_params(self, infile="srom_params.txt", delimiter=' '):
        """
        Load SROM parameters from file.

        :param infile: input file name containing SROM parameters
        :type infile: string
        :param delimiter: delimiter used in input file (default - whitespace)
        :type delimiter: string

        Returns: None. Sets sample/probability member variables.

        Assumes input file has the following format (samples in each row with
        prob after):

        | x_1^(1),   x_2^(1), ..., x_d^(1),  p^(1)
        | x_1^(2), x_2^(2), ..., x_d^(2),  p^(2)
        | ...     ...   ...    ....   ...
        | x_1^(m), x_2^(m),  ...     x_d^(m),  p^(m)

        The dimension of the samples and probabilities arrays must be
        compatible with the SROM size and dimension that was used to initialize
        the SROM class.
        """

        if not os.path.isfile(infile):
            raise IOError("SROM parameter input file does not exist: " + infile)

        srom_params = np.genfromtxt(infile, delimiter=delimiter)

        (size, dim) = srom_params.shape
        dim -= 1                        # Account for probabilities in last col.

        if size != self._size and dim != self._dim:
            msg = "Dimension mismatch when loading SROM params from file"
            raise ValueError(msg)

        self.samples = srom_params[:, :-1]
        self.probabilities = srom_params[:, -1]
