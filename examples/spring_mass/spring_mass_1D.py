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

from model import SpringMass1D
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate
from SROMPy.target import SampleRandomVector, BetaRandomVariable

# Random variable for spring stiffness.
stiffness_random_variable = \
    BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

# Specify spring-mass system:
m = 1.5                             # Deterministic mass.
state0 = [0., 0.]                   # Initial conditions.
t_grid = np.arange(0., 10., 0.1)    # Time.

# Initialize model,
model = SpringMass1D(m, state0, t_grid)

# ----------Monte Carlo------------------

# Generate stiffness input samples for Monte Carlo.
num_samples = 5000
stiffness_samples = stiffness_random_variable.draw_random_sample(num_samples)

# Calculate maximum displacement samples using MC simulation.
displacement_samples = np.zeros(num_samples)
for i, stiff in enumerate(stiffness_samples):
    displacement_samples[i] = model.get_max_disp(stiff)

# Get Monte carlo solution as a sample-based random variable:
monte_carlo_solution = SampleRandomVector(displacement_samples)

# -------------SROM-----------------------

# Generate SROM for random stiffness.
srom_size = 10
dim = 1
input_srom = SROM(srom_size, dim)
input_srom.optimize(displacement_samples)

# Compare SROM vs target stiffness distribution:
pp_input = Postprocessor(input_srom, displacement_samples)
pp_input.compare_cdfs()

# Run model to get max displacement for each SROM stiffness sample.
srom_displacements = np.zeros(srom_size)
(samples, probabilities) = input_srom.get_params()
for i, stiff in enumerate(samples):
    srom_displacements[i] = model.get_max_disp(stiff)
 
# Form new SROM for the max disp. solution using samples from the model.
output_srom = SROM(srom_size, dim)
output_srom.set_params(srom_displacements, probabilities)

# Compare solutions.
pp_output = Postprocessor(output_srom, monte_carlo_solution)
pp_output.compare_cdfs()

# --------Piecewise LINEAR surrogate with gradient info-------

# Need to calculate gradient of output wrt input samples first.

# Perturbation size for finite difference.
step_size = 1e-12
samples_fd = FD.get_perturbed_samples(samples, perturbation_values=[step_size])

# Run model to get perturbed outputs for FD calc.
perturbed_displacements = np.zeros(srom_size)
for i, stiff in enumerate(samples_fd):
    perturbed_displacements[i] = model.get_max_disp(stiff)
gradient = FD.compute_gradient(srom_displacements, perturbed_displacements,
                               [step_size])

surrogate_PWL = SROMSurrogate(input_srom, srom_displacements, gradient)
stiffness_samples = stiffness_random_variable.draw_random_sample(num_samples)
output_samples = surrogate_PWL.sample(stiffness_samples)
solution_PWL = SampleRandomVector(output_samples)

pp_pwl = Postprocessor(solution_PWL, monte_carlo_solution)
pp_pwl.compare_cdfs()


