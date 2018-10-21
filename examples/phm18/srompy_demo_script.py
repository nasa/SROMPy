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

import numpy
from os import path
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, SROMSurrogate
from SROMPy.target import SampleRandomVector

# Define target random vector from samples.
monte_carlo_input_samples_filename = path.join("mc_data", "input_samples_MC.txt")
monte_carlo_input_samples = numpy.genfromtxt(monte_carlo_input_samples_filename)
target_vector = SampleRandomVector(monte_carlo_input_samples)

# Define SROM and determine optimal parameters.
srom_size = 20
input_srom = SROM(size=srom_size, dim=3)
input_srom.optimize(target_vector)

# Compare the input CDFs (produces Figure 6).
post_processor = Postprocessor(input_srom, target_vector)
post_processor.compare_cdfs(variable_names=
                            [r'$y_{0}$', r'log$C$', r'$n$'])

# Run the model for each input SROM sample:
srom_results = numpy.zeros(srom_size)
(srom_samples, srom_probabilities) = input_srom.get_params()

# TODO: define model here.
model = None

if model is None:
    raise ValueError("model has not been defined.")

for i, sample in enumerate(srom_samples):
    srom_results[i] = model.evaluate(sample)

# Generate SROM surrogate for the end of life.
srom_surrogate_model = SROMSurrogate(input_srom, srom_results)

# Make random variable with MC end of life solution.
monte_carlo_results_filename = "eol_samples_MC.txt"
monte_carlo_results_samples = numpy.genfromtxt(monte_carlo_results_filename)
target_vector = SampleRandomVector(monte_carlo_results_samples)

# Compare final EOL solutions SROM vs MC:
# (produces Figure 7)
post_processor = Postprocessor(srom_surrogate_model, target_vector)
post_processor.compare_cdfs(variable_names=["EOL"])
