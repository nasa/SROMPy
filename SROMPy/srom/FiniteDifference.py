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
Finite Difference static class for calculating gradients
"""

import copy
import numpy as np


class FiniteDifference(object):
    """
    Class that contains static methods for assisting in computing gradients 
    needed to implement the piecewise-linear SROM surrogate using the finite
    difference method.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_perturbed_samples(samples, perturbation_factor=None,
                              perturbation_values=None):
        """
        Returns the perturbed SROM samples that must be run through model
        to estimate gradients with finite difference.

        input:
            samples: np array (m x d) - original input srom samples
            perturbation_factor: float - if specified, computes the
                perturbation size in each dimension as
                (max_i - min_i)*perturbation_factor.
                max/min_i are the max/min sample values in dim. i.
            perturbation_values: list of float - if specified uses the values
                in the array for perturbations in each dimension.

        -Must specify either perturbation_factor or perturbation_values

        output:
            returns perturbed_samples: np array (m*d x d)
            samples =  |  x^(1)_1 + delta_1, ..., x^(1)_d |
                       |  ...    , ...,      ...          |
                       |  x^(m)_1 + delta_1, ..., x^(m)_d |
                                    ....
                       |  x^(1)_1, ..., x^(1)_d + delta_d|
                       |  ...    , ...,      ...         |
                       |  x^(m)_1, ..., x^(m)_d + delta_d|

        """

        if perturbation_factor is None and perturbation_values is None:
            raise IOError("Must specify either perturbation_factor or "
                          "perturbation_values")

        # Initialize FD samples array.
        # Handle 1 dimension case, adjust shape:
        if len(samples.shape) == 1:
            samples.shape = (len(samples), 1)
        (srom_size, dim) = samples.shape
        fd_samples = np.zeros((srom_size*dim, dim))

        # Calculate perturbation_values from perturbation_factor if values
        # weren't specified:
        if perturbation_values is None:
            perturbation_values = np.array((dim, 1))
            for i in range(dim):
                ran = np.max(samples[:, i]) - np.min(samples[:, i])
                perturbation_values[i] = ran * perturbation_factor
        else:
            if len(perturbation_values) != dim:
                raise ValueError("Length of perturbation_values must equal "
                                 "dimension!")

        for i in range(dim):
            samples_i = copy.deepcopy(samples)
            samples_i[:, i] += perturbation_values[i]
            fd_samples[i*srom_size:(i+1)*srom_size, :] = samples_i

        return fd_samples

    @staticmethod
    def compute_gradient(outputs, perturbed_outputs, perturbation_values):
        """
        Calculates gradients based on original sample outputs,  perturbed
        sample outputs, and the size or perturbations.

        NOTE - it is being assumed here and other places that the output is
        scalar

        inputs:
         (mx1 array)
         outputs =            |  y(x^(1))|
                              |  ...     |
                              |  y(x^(m))|

         (mxd array)
         perturbed_outputs =  |  y(x^(1) + delta_1), ..., y(x^(1)+ delta_d)|
                              |  ...    , ...,      ...     |
                              |  y(x^(m) + delta_1), ..., y(x^(m)+delta_d)|

        (dx1 array)
        perturbed_values = [delta_1, ..., delta_d]

        outputs:
        (mxd array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_d |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_d |
        """

        if len(perturbed_outputs.shape) == 1:
            perturbed_outputs.shape = (len(perturbed_outputs), 1)
        (srom_size, dim) = perturbed_outputs.shape

        if len(outputs) != srom_size:
            raise ValueError("# output samples must match perturbed outputs")

        if len(perturbation_values) != dim:
            raise ValueError("length of perturbation_values must match "
                             "dimension")

        gradients = np.zeros((srom_size, dim))

        for i in range(dim):
            grad = ((perturbed_outputs[:, i] - outputs.flatten())
                    / perturbation_values[i])
            gradients[:, i] = grad

        return gradients

