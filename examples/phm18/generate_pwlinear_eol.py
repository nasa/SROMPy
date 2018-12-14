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
from SROMPy.srom import SROM, SROMSurrogate, FiniteDifference as FD
from SROMPy.target import SampleRandomVector

'''
Script to generate piecewise linear SROM approximation to EOL and compare it 
with the Monte Carlo solution - step 3. Uses the stored EOL model outputs 
from step 2 and the stored input SROM from step 1. In LINEAR case, need to
estimate gradients with finite difference. To do so, the model was run with
perturbed values of the inputs and the resulting EOLs were stored in the files
named "srom_fd_eol_m<>.txt". 
'''

# Monte Carlo sample data:
monte_carlo_end_of_life_sample_filename = "mc_data/eol_samples_MC.txt"
monte_carlo_end_of_life_input_filename = "mc_data/input_samples_MC.txt"

dim = 3
srom_size = 20

# Data files for EOL samples, EOL finite difference samples, and SROM inputs.
srom_end_of_life_filename = "srom_data/srom_eol_m" + \
                            str(srom_size) + ".txt"

srom_fd_end_of_life_filename = "srom_data/srom_fd_eol_m" + \
                               str(srom_size) + ".txt"

srom_input_file = "srom_data/srom_m" + str(srom_size) + ".txt"

# Get MC input/EOL samples.
monte_carlo_inputs = np.genfromtxt(monte_carlo_end_of_life_input_filename)
monte_carlo_end_of_life_data = \
    np.genfromtxt(monte_carlo_end_of_life_sample_filename)

# Get SROM EOL samples, FD samples and input SROM from file.
srom_end_of_life_data = np.genfromtxt(srom_end_of_life_filename)
srom_fd_end_of_life_data = np.genfromtxt(srom_fd_end_of_life_filename)
input_srom = SROM(srom_size, dim)
input_srom.load_params(srom_input_file)

# Get FD step sizes from file (the same for all samples, just pull the first)
# Step sizes chosen as approximately 2% of the median sample value of inputs
step_sizes = [0.0065, 0.083, 0.025]

# Calculate gradient from FiniteDifference class:
gradient = FD.compute_gradient(srom_end_of_life_data, srom_fd_end_of_life_data,
                               step_sizes)

# Create SROM surrogate, sample, and create random variable solution.
surrogate_PWL = SROMSurrogate(input_srom, srom_end_of_life_data, gradient)
srom_end_of_life_samples = surrogate_PWL.sample(monte_carlo_inputs)
solution_PWL = SampleRandomVector(srom_end_of_life_samples)

# Store EOL samples for plotting later:
end_of_life_filename = "srom_data/srom_eol_samples_m" + str(srom_size) + ".txt"
# np.savetxt(end_of_life_filename, srom_eol_samples)
# NOTE - avoid overwriting paper data

# Make MC random variable solution.
end_of_life_monte_carlo = SampleRandomVector(monte_carlo_end_of_life_data)

# Compare solutions.
pp = Postprocessor(solution_PWL, end_of_life_monte_carlo)
pp.compare_cdfs()
