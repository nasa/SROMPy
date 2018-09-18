import numpy as np

from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM, SROMSurrogate, FiniteDifference as FD
from SROMPy.target import SampleRandomVector

'''
Script to generate piecewise linear SROM approximation to EOL and compare it 
with the Monte Carlo solution - step 3. Uses the stored EOL model outputs 
from step 2 and the stored input SROM from step 1. In LINEAR case, need to
estimate gradients with finite difference. To do so, the model was run with
perturbed values of the inputs and the resulting EOLs were stored in the files
named "srom_fd_eol_m<>.txt". 
'''

#Monte Carlo sample data:
mc_eol_file = "mc_data/eol_samples_MC.txt"
mc_input_file = "mc_data/input_samples_MC.txt"

dim = 3
sromsize = 20

#Data files for EOL samples, EOL finite difference samples, and SROM inputs
srom_eol_file = "srom_data/srom_eol_m" + str(sromsize) + ".txt"
srom_fd_eol_file = "srom_data/srom_fd_eol_m" + str(sromsize) + ".txt"
srom_input_file = "srom_data/srom_m" + str(sromsize) + ".txt"

#Get MC input/EOL samples
MC_inputs = np.genfromtxt(mc_input_file)
MC_eols = np.genfromtxt(mc_eol_file)

#Get SROM EOL samples, FD samples and input SROM from file
srom_eols = np.genfromtxt(srom_eol_file)
srom_fd_eols = np.genfromtxt(srom_fd_eol_file)
input_srom  = SROM(sromsize, dim)
input_srom.load_params(srom_input_file)

#Get FD step sizes from file (the same for all samples, just pull the first)
#Step sizes chosen as approximately 2% of the median sample value of inputs
stepsizes = [0.0065, 0.083, 0.025]

#Calculate gradient from FiniteDifference class:
gradient = FD.compute_gradient(srom_eols, srom_fd_eols, stepsizes)

#Create SROM surrogate, sample, and create random variable solution
surrogate_PWL = SROMSurrogate(input_srom, srom_eols, gradient)
srom_eol_samples = surrogate_PWL.sample(MC_inputs)
solution_PWL = SampleRandomVector(srom_eol_samples)

#Store EOL samples for plotting later:
eolfile = "srom_data/srom_eol_samples_m" + str(sromsize) + ".txt"
#np.savetxt(eolfile, srom_eol_samples)  #NOTE - avoid overwriting paper data

#Make MC random variable solution
eol_mc = SampleRandomVector(MC_eols)

#COmpare solutions
pp = Postprocessor(solution_PWL, eol_mc)
pp.compare_CDFs()
