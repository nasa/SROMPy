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
This is a simple example meant to demonstrate SROMPy functionality. Estimates
the maximum displacement of a spring-mass system with a random stiffness using
SROMs and compares the solution to Monte Carlo simulation. This example is
explained in more detail in the report:

Warner, J. E. (2018). Stochastic reduced order models with Python (SROMPy). NASA/TM-2018-219824.

Note there are minor differences here due to updates in the SROMPy module
'''

import numpy as np

from spring_mass_model import SpringMassModel
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate
from SROMPy.target import SampleRandomVector, BetaRandomVariable

np.random.seed(1)

# Random variable for spring stiffness
stiffness_random_variable = \
    BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

# Specify spring-mass system:
m = 1.5                             # Deterministic mass.
state0 = [0., 0.]                   # Initial conditions.
time_step = 0.01

# Initialize model,
model = SpringMassModel(m, state0=state0, time_step=time_step)

# -------------SROM-----------------------

print "Generating SROM for input (stiffness)..."

# Generate SROM for random stiffness.
srom_size = 10
dim = 1
input_srom = SROM(srom_size, dim)
input_srom.optimize(stiffness_random_variable)

print "Computing piecewise constant SROM approximation for output (max disp)..."

# Run model to get max displacement for each SROM stiffness sample.
srom_displacements = np.zeros(srom_size)
(samples, probabilities) = input_srom.get_params()
for i, stiff in enumerate(samples):
    srom_displacements[i] = model.evaluate([stiff])
 
# Form new SROM for the max disp. solution using samples from the model.
output_srom = SROM(srom_size, dim)
output_srom.set_params(srom_displacements, probabilities)

# Compare solutions.

#Compare mean estimates for output:
print "This will be one reference solution for testing the PWC:"
print "SROM mean estimate: ", output_srom.compute_moments(1)[0][0]

# --------Piecewise LINEAR surrogate with gradient info-------

# Need to calculate gradient of output wrt input samples first.

# Perturbation size for finite difference.
step_size = 1e-12
samples_fd = FD.get_perturbed_samples(samples, perturbation_values=[step_size])

print "Computing piecewise linear SROM approximation for output (max disp)..."

# Run model to get perturbed outputs for FD calc.
perturbed_displacements = np.zeros(srom_size)
for i, stiff in enumerate(samples_fd):
    perturbed_displacements[i] = model.evaluate([stiff])

gradient = FD.compute_gradient(srom_displacements, perturbed_displacements,
                               [step_size])

surrogate_PWL = SROMSurrogate(input_srom, srom_displacements, gradient)

print "This will be another reference solution for testing the PWL:"
print "surrogatePWL(2) = ", surrogate_PWL.sample(np.array([2]))



