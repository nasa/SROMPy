import numpy as np

from model import SpringMass_1D
from src.postprocess import Postprocessor
from src.srom import SROM, FiniteDifference as FD, SROMSurrogate
from src.target import SampleRandomVector, BetaRandomVariable

#Random variable for spring stiffness
stiffness_rv = BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

#Specify spring-mass system:
m = 1.5                             #deterministic mass
state0 = [0., 0.]                   #initial conditions
t_grid = np.arange(0., 10., 0.1)    #time 

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
mc_solution = SampleRandomVector(disp_samples)

#-------------SROM-----------------------

#generate SROM for random stiffness
sromsize = 10
dim = 1
input_srom = SROM(sromsize, dim)
input_srom.optimize(stiffness_rv)

#Compare SROM vs target stiffness distribution:
pp_input = Postprocessor(input_srom, stiffness_rv)
pp_input.compare_CDFs()

#Run model to get max disp for each SROM stiffness sample
srom_disps = np.zeros(sromsize)
(samples, probs) = input_srom.get_params()
for i, stiff in enumerate(samples):
    srom_disps[i] = model.get_max_disp(stiff)
 
#Form new SROM for the max disp. solution using samples from the model   
output_srom = SROM(sromsize, dim)
output_srom.set_params(srom_disps, probs)

#Compare solutions
pp_output = Postprocessor(output_srom, mc_solution)
pp_output.compare_CDFs()

#--------Piecewise LINEAR surrogate with gradient info-------

#Need to calculate gradient of output wrt input samples first 

#Perturbation size for finite difference
stepsize = 1e-12
samples_fd = FD.get_perturbed_samples(samples, perturb_vals=[stepsize])

#Run model to get perturbed outputs for FD calc.
perturbed_disps = np.zeros(sromsize)
for i, stiff in enumerate(samples_fd):
    perturbed_disps[i] = model.get_max_disp(stiff)
gradient = FD.compute_gradient(srom_disps, perturbed_disps, [stepsize])

surrogate_PWL = SROMSurrogate(input_srom, srom_disps, gradient)
stiffness_samples = stiffness_rv.draw_random_sample(num_samples)
output_samples = surrogate_PWL.sample(stiffness_samples)
solution_PWL = SampleRandomVector(output_samples)

pp_pwl = Postprocessor(solution_PWL, mc_solution)
pp_pwl.compare_CDFs()


