import numpy as np

#from model import Model (assume this exists)
from src.postprocess import Postprocessor
from src.srom import SROM, SROMSurrogate
from src.target import SampleRandomVector

'''
Generate SROM to model input distribution (samples)
'''

#Specify input/output files and SROM optimization parameters
dim = 3
srom_size = 20
mc_input_file = "mc_data/input_samples_MC.txt"
mc_eol_file = "mc_data/eol_samples_MC.txt"

#Define target random variable from samples
MCsamples = np.genfromtxt(samplesfile)
target = SampleRandomVector(MCsamples)

#Define SROM, determine optimal parameters, store parameters
input_srom = SROM(srom_size, dim)
input_srom.optimize(target, weights=[1,1,1], error="SSE")

#Compare the CDFs
pp = Postprocessor(srom, target)
pp.compare_CDFs(saveFig=False)

#Run the model for each input SROM sample:
srom_eols = np.zeros(srom_size)
(srom_samples, srom_probs) = input_srom.get_params()
for i, sample in enumerate(srom_samples):
    srom_eols[i] = model.evaluate(sample)

#Generate SROM surrogate for the output
eol_srom = SROMSurrogate(input_srom, srom_eols)

#Make random variable with MC eol solution
MC_eols = np.genfromtxt(mc_eol_file)
eol_mc = SampleRandomVector(MC_eols)

#Compare final EOL solutions SROM vs MC:
pp = Postprocessor(eol_srom, eol_mc)
pp.compare_CDFs()

