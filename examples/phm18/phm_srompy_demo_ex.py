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

#from model import Model (assume this exists)
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, SROMSurrogate
from SROMPy.target import SampleRandomVector

'''
Generate SROM to model input distribution (samples)
'''

# Specify input/output files and SROM optimization parameters.
dim = 3
srom_size = 20
monte_carlo_end_of_life_input_filename = "mc_data/input_samples_MC.txt"
monte_carlo_end_of_life_sample_filename = "mc_data/eol_samples_MC.txt"

# Define target random variable from samples.
monte_carlo_samples = np.genfromtxt(monte_carlo_end_of_life_input_filename)
target = SampleRandomVector(monte_carlo_samples)

# Define SROM, determine optimal parameters, store parameters.
input_srom = SROM(srom_size, dim)
input_srom.optimize(target, weights=[1, 1, 1], error="SSE")

# Compare the CDFs.
pp = Postprocessor(input_srom, target)
pp.compare_cdfs(save_figure=False)

# Run the model for each input SROM sample:
srom_end_of_life_data = np.zeros(srom_size)
(srom_samples, srom_probabilities) = input_srom.get_params()
for i, sample in enumerate(srom_samples):
    srom_end_of_life_data[i] = model.evaluate(sample)

# Generate SROM surrogate for the output.
eol_srom = SROMSurrogate(input_srom, srom_end_of_life_data)

# Make random variable with MC eol solution.
monte_carlo_eols = np.genfromtxt(monte_carlo_end_of_life_sample_filename)
eol_mc = SampleRandomVector(monte_carlo_eols)

# Compare final EOL solutions SROM vs MC:
pp = Postprocessor(eol_srom, eol_mc)
pp.compare_cdfs()

