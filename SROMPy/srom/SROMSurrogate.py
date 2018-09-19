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

'''
Define SROM-based output surrogate class
'''

import numpy as np

from SROM import SROM

class SROMSurrogate:
    """
    SROMPy class that provides a closed-form surrogate model for a model output
    that can be sampled as a means of efficiently propagating uncertainty.
    Enables both a piecewise-constant model and a piecewise-linear model, if
    gradient information is provided. 

    :param inputsrom: The input SROM that was used to generate the outputs.
    :type inputsrom: SROMPy SROM object.
    :param outputsamples: Output samples corresponding to each input SROM sample
    :type outputsamples: 2d Numpy Array
    :param outputgradients: Gradient of output with respect to input samples
    :type outputgradients: 2d Numpy Array
    
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

    def __init__(self, inputsrom, outputsamples, outputgradients=None):
        '''
        Initialize SROM surrogate using the input SROM used to generate the
        output samples. Output gradients are also supplied for the case of
        the piecewise linear surrogate.

        output samples have the following convention:

        m = SROM size (superscript), do = dimension (subscript);

        outputsamples =  |  y^(1)_1, ..., y^(1)_do |
                         |  ...    , ..., ...      |
                         |  y^(m)_1, ..., y^(m)_do |

        output samples must match the order of inputsrom samples/probs!

        (m x d_i array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_di |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_di |

        do - dimension of output samples (doesn't need to equal di of input)

        '''

        if inputsrom._samples is None or inputsrom._probs is None:
            raise ValueError("Input SROM must be properly initialized")

        self._inputsrom = inputsrom

        #Handle 1 dimension case, adjust shape:
        if len(outputsamples.shape) == 1:
            outputsamples.shape = (len(outputsamples), 1)

        #Verify dimensions of samples/probs
        (size, dim) = outputsamples.shape

        if size != self._inputsrom._size:
            raise ValueError("Number of output samples must match input " +
                             " srom size!")

        self._outsamples = outputsamples
        self._dim = dim
        self._size = size

        #TODO - checks on outputgradients:
        if outputgradients is not None:
            (size__, dim__) = outputgradients.shape
            if size__ != self._inputsrom._size:
                raise ValueError("Incorrect # samples in gradient array!")
            if dim__ != self._inputsrom._dim:
                raise ValueError("Incorrect dimension in gradient array!")

        self._gradients = outputgradients

        #Make SROM for output?
        self._outputsrom = SROM(size, dim)
        self._outputsrom.set_params(outputsamples, inputsrom._probs)


    #Do these change for linear surrogate?
    def compute_moments(self, max_order):
        """
        Calculates and returns SROM moments.

        :param max_order: Maximum order of moments to return
        :type max_order: int

        Returns (max_order x dim) size Numpy array with SROM moments for 
        each dimension.
        """

        return self._outputsrom.compute_moments(max_order)

    def compute_CDF(self, x_grid):
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

        return self._outputsrom.compute_CDF(x_grid)

    def sample(self, inputsamples):
        '''
        Generates output samples from the SROM surrogate corresponding to
        the provided input samples.

        :param inputsamples: samples of inputs to draw output samples for
        :type inputsamples: 2d Numpy array.

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

        '''

        #Handle 1 dimension case, adjust shape:
        if len(inputsamples.shape) == 1:
            inputsamples.shape = (len(inputsamples), 1)

        #Verify dimensions of samples/probs
        (numsamples, dim) = inputsamples.shape

        if dim != self._inputsrom._dim:
            raise ValueError("Incorrect input sample dimension")

        #Evaluate piecewise constant or linear surrogate model to get samples:
        if self._gradients is None:
            surr_samples = self._sample_pwconstant_surrogate(inputsamples)
        else:
            surr_samples = self._sample_pwlinear_surrogate(inputsamples)

        return surr_samples

    def _sample_pwconstant_surrogate(self, inputsamples):
        '''
        Evaluate standard piecewise constant output surrogate model
        '''

        inputsamples_srom = self._inputsrom._samples

        #Generate surrogate samples:
        (numsamples, _) = inputsamples.shape

        #Generate surrogate samples:
        surr_samples = np.zeros((numsamples, self._dim))
        for i in range(numsamples):
            #Find which input SROM sample is closest to current sample
            sample_i = inputsamples[i, :]
            diff_norms = np.linalg.norm(sample_i - inputsamples_srom, axis=1)
            sromindex = np.argmin(diff_norms)
            surr_samples[i, :] = self._outsamples[sromindex, :]

        return surr_samples

    def _sample_pwlinear_surrogate(self, inputsamples): 
        '''
        Evaluate the linear output surrogate model using input SROM samples
        and gradients

        input:
        inputsamples =  |  x^(1)_1, ..., x^(1)_di |
                        |  ...    , ..., ...      |
                        |  x^(N)_1, ..., x^(N)_di |

        (mxd array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_d |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_d |

        '''

        inputsamples_srom = self._inputsrom._samples

        #Generate surrogate samples:
        (numsamples, _) = inputsamples.shape

        #Generate surrogate samples:
        surr_samples = np.zeros((numsamples, self._dim))
        for i in range(numsamples):
            #Find which input SROM sample is closest to current sample
            sample_i = inputsamples[i, :]
            diffs = sample_i - inputsamples_srom
            diff_norms = np.linalg.norm(diffs, axis=1)
            sromindex = np.argmin(diff_norms)

            #Calculate ouput sample value (eq 11b from emery paper)
            output_k = self._outsamples[sromindex, :]
            diffs_k = diffs[sromindex, :]
            grad_k = self._gradients[sromindex, :]

            out = output_k + np.dot(grad_k, diffs_k)
            surr_samples[i, :] = out

        return surr_samples

