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

import numpy as np

from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector

'''
Generate SROM to model input distribution (samples)
'''

# Specify input/output files and SROM optimization parameters.
dim = 3
srom_size = 20
samples_file = "mc_data/input_samples_MC.txt"
outfile = "srom_data/srom_m" + str(srom_size) + ".txt"

# Define target random variable from samples
mc_samples = np.genfromtxt(samples_file)
target = SampleRandomVector(mc_samples)

# Define SROM, determine optimal parameters, store parameters
srom = SROM(srom_size, dim)
srom.optimize(target, weights=[1, 1, 1], error="SSE", num_test_samples=100)

# NOTE - commented out to not overwrite paper data files:
# srom.save_params(outfile)

# Check out the CDFs.
pp = Postprocessor(srom, target)
pp.compare_cdfs(variable_names=[r'$y_{0}$', r'log$C$', r'$n$'])

