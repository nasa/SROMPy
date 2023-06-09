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
Define SROM-based output surrogate class
"""

import numpy as np

from .SROM import SROM


class SROMSurrogate:
    """
    SROMPy class that provides a closed-form surrogate model for a model output
    that can be sampled as a means of efficiently propagating uncertainty.
    Enables both a piecewise-constant model and a piecewise-linear model, if
    gradient information is provided.
    
    Conventions:

    * m denotes the SROM size (superscripts). di denotes the dimension of the 
      SROM input (subscripts). do denotes dimension of SROM output (subscripts).
    * The output samples array has the following layout (m x d0):

        | [[ y^(1)_1,   y_2^(1), ..., y_do^(1)],
        | [y_1^(2), y_2^(2), ..., y_d0^(2)],
        | ...     ...   ...    ....  
        | [y_1^(m), y_2^(m),  ...  y_d0^(m)]]    
    
    * The gradients array has the following layout (m x di):

        | [[dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_di ],
        | ...             , ...,    ...           
        | [dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_di ]]

    Note

    * the order of the output samples array must match the order of the
      samples array from the input SROM!
    * If gradients array is provided, the piecewise-linear surrogate 
      model is implemented. Otherwise, the piecewise-constant surrogate is
      used.

    """

    def __init__(self, input_srom, output_samples, output_gradients=None):
        """
        Initialize SROM surrogate using the input SROM used to generate the
        output samples. Output gradients are also supplied for the case of
        the piecewise linear surrogate.

        output samples have the following convention:

        m = SROM size (superscript), do = dimension (subscript);

        output_samples =  |  y^(1)_1, ..., y^(1)_do |
                         |  ...    , ..., ...      |
                         |  y^(m)_1, ..., y^(m)_do |

        output samples must match the order of input_srom samples/probabilities!

        (m x d_i array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_di |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_di |

        do - dimension of output samples (doesn't need to equal di of input)

        :param input_srom: The input SROM that was used to generate the outputs.
        :type input_srom: SROMPy SROM object.
        :param output_samples: Output samples corresponding to each input SROM
            sample
        :type output_samples: 2d Numpy Array
        :param output_gradients: Gradient of output with respect to input
            samples
        :type output_gradients: 2d Numpy Array
        """

        if input_srom.samples is None or input_srom.probabilities is None:
            raise ValueError("Input SROM must be properly initialized")

        self._input_srom = input_srom

        # Handle 1 dimension case, adjust shape:
        if len(output_samples.shape) == 1:
            output_samples.shape = (len(output_samples), 1)

        # Verify dimensions of samples/probabilities.
        (size, dim) = output_samples.shape

        if size != self._input_srom._size:
            raise ValueError("Number of output samples must match input " +
                             " srom size!")

        self._out_samples = output_samples
        self._dim = dim
        self._size = size

        # TODO - checks on output_gradients:
        if output_gradients is not None:

            (size__, dim__) = output_gradients.shape
            if size__ != self._input_srom._size:
                raise ValueError("Incorrect # samples in gradient array!")
            if dim__ != self._input_srom._dim:
                raise ValueError("Incorrect dimension in gradient array!")

        self._gradients = output_gradients

        # Make SROM for output?
        self._output_srom = SROM(size, dim)
        self._output_srom.set_params(output_samples, input_srom.probabilities)

    # Do these change for linear surrogate?
    def compute_moments(self, max_order):
        """
        Calculates and returns SROM moments.

        :param max_order: Maximum order of moments to return
        :type max_order: int

        Returns (max_order x dim) size Numpy array with SROM moments for 
        each dimension.
        """

        return self._output_srom.compute_moments(max_order)

    def compute_cdf(self, x_grid):
        """
        Computes the SROM marginal CDF values in each dimension.

        :param x_grid: Grid of points to compute CDF values on. If 1d array is
            provided, the same points are used to evaluate CDF in each 
            dimension. If 2d array is provided, calculates CDF values on
            different points, but must have same # points for each dimension. 
            Size is (# grid pts) x (dim) or (# grid pts) x (1).
        :type x_grid: Numpy array.

        Returns: Numpy array of CDF values at x_grid points. Size is (# grid 
        pts) x (dim).

        Note: 
            * Increasing the number of grid points can significantly slow 
              down the SROM optimization problem.
            * Providing a 2d array for x_grid can specify a different range
              of values for each dimension, but must use the same number of pts.
        """

        return self._output_srom.compute_cdf(x_grid)

    def sample(self, input_samples):
        """
        Generates output samples from the SROM surrogate corresponding to
        the provided input samples.

        :param input_samples: samples of inputs to draw output samples for
        :type input_samples: 2d Numpy array.

        Returns: 2d Numpy array of output samples corresponding to input samples

        Convention:
            * N - number of samples. di - dimension of the input. do - dimension
              of the output. 
            * input samples array has following layout (N x di):


              | [[x^(1)_1, ..., x^(1)_di ],    
              |  ...    , ..., ...   
              | [x^(N)_1, ..., x^(N)_di ]]   
        
            * surrogate output samples has following layout (N x do):

              | [[y^(1)_1, ..., y^(1)_do ],    
              |  ...    , ..., ...      
              | [y^(N)_1, ..., y^(N)_do ]]   

        Note that the samples are drawn from a piecewise-linear SROM 
        surrogate when gradients are provided to the constructor of this class,
        and drawn from a piecewise-constant SROM surrogate if not.

        """

        # Handle 1 dimension case, adjust shape:
        if len(input_samples.shape) == 1:
            input_samples.shape = (len(input_samples), 1)

        # Verify dimensions of samples/probabilities.
        (num_samples, dim) = input_samples.shape

        if dim != self._input_srom._dim:
            raise ValueError("Incorrect input sample dimension")

        # Evaluate piecewise constant or linear surrogate model to get samples:
        if self._gradients is None:
            surrogate_samples = \
                self._sample_piecewise_constant_surrogate(input_samples)
        else:
            surrogate_samples = \
                self._sample_piecewise_linear_surrogate(input_samples)

        return surrogate_samples

    def _sample_piecewise_constant_surrogate(self, input_samples):
        """
        Evaluate standard piecewise constant output surrogate model
        """

        input_samples_srom = self._input_srom.samples

        # Generate surrogate samples:
        (num_samples, _) = input_samples.shape

        # Generate surrogate samples:
        surrogate_samples = np.zeros((num_samples, self._dim))
        for i in range(num_samples):

            # Find which input SROM sample is closest to current sample.
            sample_i = input_samples[i, :]
            diff_norms = np.linalg.norm(sample_i - input_samples_srom, axis=1)
            srom_index = np.argmin(diff_norms)
            surrogate_samples[i, :] = self._out_samples[srom_index, :]

        return surrogate_samples

    def _sample_piecewise_linear_surrogate(self, input_samples):
        """
        Evaluate the linear output surrogate model using input SROM samples
        and gradients

        input:
        input_samples =  |  x^(1)_1, ..., x^(1)_di |
                        |  ...    , ..., ...      |
                        |  x^(N)_1, ..., x^(N)_di |

        (mxd array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_d |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_d |

        """

        input_samples_srom = self._input_srom.samples

        # Generate surrogate samples:
        (num_samples, _) = input_samples.shape

        # Generate surrogate samples:
        surrogate_samples = np.zeros((num_samples, self._dim))
        for i in range(num_samples):

            # Find which input SROM sample is closest to current sample.
            sample_i = input_samples[i, :]
            diffs = sample_i - input_samples_srom
            diff_norms = np.linalg.norm(diffs, axis=1)
            srom_index = np.argmin(diff_norms)

            # Calculate output sample value (eq 11b from emery paper).
            output_k = self._out_samples[srom_index, :]
            diffs_k = diffs[srom_index, :]
            grad_k = self._gradients[srom_index, :]

            out = output_k + np.dot(grad_k, diffs_k)
            surrogate_samples[i, :] = out

        return surrogate_samples
