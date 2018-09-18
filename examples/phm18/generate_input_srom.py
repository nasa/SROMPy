import numpy as np

from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector

'''
Generate SROM to model input distribution (samples)
'''

#Specify input/output files and SROM optimization parameters
dim = 3
srom_size = 20
samplesfile = "mc_data/input_samples_MC.txt"
outfile = "srom_data/srom_m" + str(srom_size) + ".txt"

#Define target random variable from samples
MCsamples = np.genfromtxt(samplesfile)
target = SampleRandomVector(MCsamples)

#Define SROM, determine optimal parameters, store parameters
srom = SROM(srom_size, dim)
srom.optimize(target, weights=[1,1,1], error="SSE", num_test_samples=100)

#NOTE - commented out to not overwrite paper data files:
#srom.save_params(outfile)

#Check out the CDFs
pp = Postprocessor(srom, target)
pp.compare_CDFs(variablenames=[r'$y_{0}$', r'log$C$', r'$n$'])

