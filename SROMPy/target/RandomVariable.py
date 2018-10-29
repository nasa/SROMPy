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

import abc
from SROMPy.target.RandomEntity import RandomEntity

"""
Abstract class defining the target random variable being matched by an SROM.
Inherited by BetaRandomVariable and GammaRandomVariable, 
and NormalRandomVariable.
"""


class RandomVariable(RandomEntity):

    @abc.abstractmethod
    def get_variance(self, max_order):
        return
    
    @abc.abstractmethod
    def compute_moments(self, x_grid):
        return

    @abc.abstractmethod
    def compute_cdf(self):
        return

    @abc.abstractmethod
    def compute_inv_cdf(self, sample_size):
        return

    @abc.abstractmethod
    def compute_pdf(self):
        return

    @abc.abstractmethod
    def draw_random_sample(self):
        return

    @abc.abstractmethod
    def generate_moments(self):
        return
