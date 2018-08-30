import numpy as np

from src.srom import SROM

'''
Run computational model for each input SROM sample - step 2
NOTE - this script will not run, "Model" class is not provided. But this 
script is representative of a common SROM workflow.
'''

dim = 3
srom_size = 20
sromfile = "srom_data/srom_m" + str(srom_size) + ".txt"
sromeolfile = "srom_data/srom_eol_m" + str(srom_size) + ".txt"

srom = SROM(srom_size, dim)
srom.load_params(sromfile)

srom_outputs = np.zeros(srom_size)
(srom_samples, srom_probs) = srom.get_params()
for i, input in enumerate(srom_samples):
    srom_outputs[i] = model.evaluate(input)

np.savetxt(sromeolfile, srom_outputs)
