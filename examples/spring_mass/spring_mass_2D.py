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

from model import SpringMass2D
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import BetaRandomVariable as beta
from SROMPy.target import SampleRandomVector

#Specify spring-mass system:
state0 = [0., 0.]                   #initial conditions
t_grid = np.arange(0., 10., 0.1)    #time 

#random variable for spring stiffness & mass
stiffness_rv = beta(alpha=3., beta=2., shift=1., scale=2.5)
mass_rv = beta(alpha=2./3., beta=1./3., shift=0.5, scale=1.5)

#Initialize model
model = SpringMass2D(state0, t_grid)

#Generate samples of random variables for sample based random vector
k_samples = stiffness_rv.draw_random_sample(5000)
m_samples = mass_rv.draw_random_sample(5000)
km_samples = np.array([k_samples, m_samples]).T

#----------Monte Carlo------------------

#Generate stiffness input samples for Monte Carlo
num_samples = 5000

k_samples = stiffness_rv.draw_random_sample(num_samples)
m_samples = mass_rv.draw_random_sample(num_samples)

#Calculate maximum displacement samples using MC simulation
disp_samples = np.zeros(num_samples)
for i in range(num_samples):
    disp_samples[i] = model.get_max_disp(k_samples[i], m_samples[i])

#Get Monte carlo solution as a sample-based random variable:
mc_solution = SampleRandomVector(disp_samples)

#-------------SROM-----------------------

#generate SROM for random vector of stiffness & mass
sromsize = 25
dim = 2

#Assume we only have access to samples in this example and want SROM from them:
km_samples = np.array([k_samples, m_samples]).T
km_random_vector = SampleRandomVector(km_samples)

srom = SROM(sromsize, dim)
srom.optimize(km_random_vector)
(samples, probs) = srom.get_params()

#Run model to get max disp for each SROM stiffness sample
srom_disps = np.zeros(sromsize)
for i in range(sromsize):
    k = samples[i,0]
    m = samples[i,1]
    srom_disps[i] = model.get_max_disp(k, m)
 
#Form new SROM for the max displacement solution using samples from the model   
srom_solution = SROM(sromsize, 1)
srom_solution.set_params(srom_disps, probs)

#----------------------------------------

#Compare solutions
pp = Postprocessor(srom_solution, mc_solution)
pp.compare_cdfs()


