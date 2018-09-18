import numpy as np

from SROMPy.srom import SROM

from Model import CrackGrowthModel

'''
Run computational model for each input SROM sample - step 2
NOTE - this script will not run, Model class is not provided. But this 
script is representative of a common SROM workflow.

First, we load the SROM parameters that were generated in step 1, then we
get the samples for that SROM and evaluate the crack growth model for each one, 
and store the outputs (EOL) from the model.
'''

#Initialize crack growth model (not provided)
model = CrackGrowthModel()

dim = 3
srom_size = 20

sromfile = "srom_data/srom_m" + str(srom_size) + ".txt"
sromeolfile = "srom_data/srom_eol_m" + str(srom_size) + ".txt"

#Initialize SROM and load parameters from file
srom = SROM(srom_size, dim)
srom.load_params(sromfile)

#Evaluate the crack growth model for each SROM input sample
srom_outputs = np.zeros(srom_size)
(srom_samples, srom_probs) = srom.get_params()
for i, input in enumerate(srom_samples):
    srom_outputs[i] = model.evaluate(input)

#Save EOL outputs for step 3:
np.savetxt(sromeolfile, srom_outputs)
