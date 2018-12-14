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

from SROMPy.srom import SROM

from Model import CrackGrowthModel

'''
Run computational model for each input SROM sample - step 2
NOTE - this script will not run, Model class is not provided. But this 
script is representative of a common SROM workflow.

First, we load the SROM parameters that were generated in step 1, then we
get the samples for that SROM and evaluate the crack growth model for each one, 
and store the outputs (EOL) from the model.
'''

# Initialize crack growth model (not provided).
model = CrackGrowthModel()

dim = 3
srom_size = 20

srom_filename = "srom_data/srom_m" + str(srom_size) + ".txt"
srom_end_of_life_filename = "srom_data/srom_eol_m" + str(srom_size) + ".txt"

# Initialize SROM and load parameters from file.
srom = SROM(srom_size, dim)
srom.load_params(srom_filename)

# Evaluate the crack growth model for each SROM input sample.
srom_outputs = np.zeros(srom_size)
(srom_samples, srom_probabilities) = srom.get_params()

for i, srom_sample in enumerate(srom_samples):
    srom_outputs[i] = model.evaluate(srom_sample)

# Save EOL outputs for step 3:
np.savetxt(srom_end_of_life_filename, srom_outputs)
