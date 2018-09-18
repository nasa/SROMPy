import numpy as np

from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import SampleRandomVector

'''
Ex3 - unimodal 3D
Script to generate PW constant SROM approximation to EOL and compare it with the
monte carlo solution (w/ surrogate model)
'''

mc_eol_file = "mc_data/eol_samples_MC.txt"

sromsize = 20

srom_eol_file = "srom_data/srom_eol_m" + str(sromsize) + ".txt"
srom_input_file = "srom_data/srom_m" + str(sromsize) + ".txt"

#Get MC EOL samples
MC_eols = np.genfromtxt(mc_eol_file)

#Get SROM EOL samples & probabilities from input srom
srom_eols = np.genfromtxt(srom_eol_file)
srom_probs = np.genfromtxt(srom_input_file)[:,-1]  #probs in last column

#Make MC random variable & SROM to compare
eol_srom = SROM(sromsize, dim=1)
eol_srom.set_params(srom_eols, srom_probs)
eol_mc = SampleRandomVector(MC_eols)

pp = Postprocessor(eol_srom, eol_mc)
pp.compare_CDFs(variablenames=["EOL"])

