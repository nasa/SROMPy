
import numpy as np
import matplotlib.pyplot as plt

from model import SpringMass_1D, SpringMass_2D
from srom import SROM
from target import BetaRandomVariable as beta


#Specify spring-mass system:
m = 1.5                             #deterministic mass
state0 = [0., 0.]                   #initial conditions
t_grid = np.arange(0., 10., 0.1)    #time 

#random variable for spring stiffness
stiffness_rv = beta(alpha=3., beta=2., shift=1., scale=2.5)

#Initialize model
model = SpringMass_1D(m, state0, t_grid)

#Generate stiffness input samples for Monte Carlo
num_samples = 5000
stiffness_samples = stiffness_rv.draw_random_sample(num_samples)

if False:
    #Calculate maximum displacement samples using MC simulation
    disp_samples = np.zeros(num_samples)
    for i, stiff in enumerate(stiffness_samples):
        disp_samples[i] = model.get_max_disp(stiff)


    plt.figure()
    plt.hist(disp_samples)
    plt.show()


#generate SROM for random stiffness
sromsize = 5
dim = 1
weights = [1.,0.,0.]
srom = SROM(sromsize, dim)
srom.optimize(stiffness_rv)


