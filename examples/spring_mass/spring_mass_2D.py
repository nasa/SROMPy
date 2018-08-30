
import numpy as np
import matplotlib.pyplot as plt

from model import SpringMass_1D, SpringMass_2D
from postprocess import Postprocessor
from srom import SROM
from target import BetaRandomVariable as beta
from target import SampleRandomVector

#Specify spring-mass system:
state0 = [0., 0.]                   #initial conditions
t_grid = np.arange(0., 10., 0.1)    #time 

#random variable for spring stiffness & mass
stiffness_rv = beta(alpha=3., beta=2., shift=1., scale=2.5)
mass_rv = beta(alpha=2./3., beta=1./3., shift=0.5, scale=1.5)

#Initialize model
model = SpringMass_2D(state0, t_grid)

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
pp.compare_CDFs()


