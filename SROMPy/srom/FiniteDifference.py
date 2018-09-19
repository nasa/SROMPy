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
Finite Difference static class for calculating gradients
'''

import copy
import numpy as np


class FiniteDifference(object):
    '''
    Class that contains static methods for assisting in computing gradients 
    needed to implement the piecewise-linear SROM surrogate using the finite
    difference method.
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_perturbed_samples(samples, perturb_fact=None, perturb_vals=None):
        '''
        Returns the perturbed SROM samples that must be run through model
        to estimate gradients with finite difference.

        input:
            samples: np array (m x d) - original input srom samples
            perturb_fact: float - if specified, computes the perturbation size
                            in each dimension as (max_i - min_i)*perturb_fact.
                            max/min_i are the max/min sample values in dim. i.
            perturb_vals: list of float - if specified uses the values in the
                            array for perturbations in each dimension.

        -Must specify either perturb_fact or perturb_vals

        output:
            returns perturbed_samples: np array (m*d x d)
            samples =  |  x^(1)_1 + delta_1, ..., x^(1)_d |
                       |  ...    , ...,      ...          |
                       |  x^(m)_1 + delta_1, ..., x^(m)_d |
                                    ....
                       |  x^(1)_1, ..., x^(1)_d + delta_d|
                       |  ...    , ...,      ...         |
                       |  x^(m)_1, ..., x^(m)_d + delta_d|

        '''

        if perturb_fact is None and perturb_vals is None:
            raise IOError("Must specify either perturb_fact or perturb_vals")

        #Initialize FD samples array
        #Handle 1 dimension case, adjust shape:
        if len(samples.shape) == 1:
            samples.shape = (len(samples), 1)
        (sromsize, dim) = samples.shape
        fd_samples = np.zeros((sromsize*dim, dim))

        #Calculate perturb_vals from perturb_fact if vals werent specified:
        if perturb_vals is None:
            perturb_vals = np.array((dim, 1))
            for i in range(dim):
                ran = np.max(samples[:, i]) - np.min(samples[:, i])
                perturb_vals[i] = ran*perturb_fact
        else:
            if len(perturb_vals) != dim:
                raise ValueError("Length of perturb_vals must equal dimension!")

        for i in range(dim):
            samples_i = copy.deepcopy(samples)
            samples_i[:, i] += perturb_vals[i]
            fd_samples[i*sromsize:(i+1)*sromsize, :] = samples_i

        return fd_samples


    @staticmethod
    def compute_gradient(outputs, perturbed_outputs, perturb_vals):
        '''
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
        perturbed_vals = [delta_1, ..., delta_d]

        outputs:
        (mxd array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_d |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_d |
        '''

        if len(perturbed_outputs.shape) == 1:
            perturbed_outputs.shape = (len(perturbed_outputs), 1)
        (sromsize, dim) = perturbed_outputs.shape

        if len(outputs) != sromsize:
            raise ValueError("# output samples must match perturbed outputs")

        if len(perturb_vals) != dim:
            raise ValueError("length of perturb_vals must match dimension")

        gradients = np.zeros((sromsize, dim))

        for i in range(dim):
            grad = ((perturbed_outputs[:, i] - outputs.flatten())
                    /perturb_vals[i])
            gradients[:, i] = grad

        return gradients

