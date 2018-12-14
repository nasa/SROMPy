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
Ex3 - unimodal 3D
Script to generate PW constant SROM approximation to EOL and compare it with the
monte carlo solution (w/ surrogate model)
'''

mc_eol_file = "mc_data/eol_samples_MC.txt"

srom_size = 20

srom_eol_file = "srom_data/srom_eol_m" + str(srom_size) + ".txt"
srom_input_file = "srom_data/srom_m" + str(srom_size) + ".txt"

# Get MC EOL samples.
MC_eols = np.genfromtxt(mc_eol_file)

# Get SROM EOL samples & probabilities from input srom.
srom_end_of_life_data = np.genfromtxt(srom_eol_file)

# Probabilities in last column.
srom_probabilities = np.genfromtxt(srom_input_file)[:, -1]

# Make MC random variable & SROM to compare.
end_of_life_srom = SROM(srom_size, dim=1)
end_of_life_srom.set_params(srom_end_of_life_data, srom_probabilities)
end_of_life_monte_carlo = SampleRandomVector(MC_eols)

pp = Postprocessor(end_of_life_srom, end_of_life_monte_carlo)
pp.compare_cdfs(variable_names=["EOL"])

