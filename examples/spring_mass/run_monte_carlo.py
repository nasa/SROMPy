
import numpy as np
import matplotlib.pyplot as plt

from model import SpringMass_1D, SpringMass_2D
from postprocess import Postprocessor
from srom import SROM
from target import BetaRandomVariable as beta
from target import SampleRV 

#Specify spring-mass system:
m = 1.5                             #deterministic mass
state0 = [0., 0.]                   #initial conditions
t_grid = np.arange(0., 10., 0.1)    #time 

#random variable for spring stiffness
stiffness_rv = beta(alpha=3., beta=2., shift=1., scale=2.5)

#Initialize model
model = SpringMass_1D(m, state0, t_grid)

#----------Monte Carlo------------------

#Generate stiffness input samples for Monte Carlo
num_samples = 5000
stiffness_samples = stiffness_rv.draw_random_sample(num_samples)

#Calculate maximum displacement samples using MC simulation
disp_samples = np.zeros(num_samples)
for i, stiff in enumerate(stiffness_samples):
    disp_samples[i] = model.get_max_disp(stiff)

#Get Monte carlo solution as a sample-based random variable:
mc_solution = SampleRV(disp_samples)

#-------------SROM-----------------------

#generate SROM for random stiffness
sromsize = 10
dim = 1

srom = SROM(sromsize, dim)
srom.optimize(stiffness_rv)
(samples, probs) = srom.get_params()

#Run model to get max disp for each SROM stiffness sample
srom_disps = np.zeros(sromsize)
for i, stiff in enumerate(samples):
    srom_disps[i] = model.get_max_disp(stiff)
    
srom_solution = SROM(sromsize, dim)
srom_solution.set_params(srom_disps, probs)

#----------------------------------------

#Compare solutions
pp = Postprocessor(srom_solution, mc_solution)
pp.compare_CDFs()


