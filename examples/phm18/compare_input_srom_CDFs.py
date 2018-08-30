import os
import numpy as np
from collections import OrderedDict

from src.target import SampleRandomVector
from src.srom import SROM
from src.postprocess import Postprocessor

'''
Compare SROMs for inputs - produce figure 3 in the paper
'''

#Target Monte Carlo input samples for comparison
targetsamples = "mc_data/input_samples_MC.txt"

#SElect 3 SROM sizes
sromsizes = [5,10,20]
srom_dir = "srom_data"

#Plotting specs:
varz = [r'log$C$', r'$y_{0}$', r'$n$']
cdfylabel = True        #Label y axis as "CDF"
plot_dir = "plots"
plot_suffix = "SROM_input_CDF_m"
for m in sromsizes:
    plot_suffix += "_" + str(m)

#Xtick labels for each variable (only modify y0):
y0ticks = ['0.245', '', '0.255', '', '0.265', '', '0.275']
y0ticks = ['', '0.245', '', '0.255', '', '0.265', '', '0.275']
xticks = [None, y0ticks, None]

#Load / initialize target random variable from samples:
samples = np.genfromtxt(targetsamples)
target = SampleRandomVector(samples)

#Set x limits for each variable based on target:
xlimits = []
for i in range(target._dim):
    lims = [np.min(samples[:,i]), np.max(samples[:,i])]
    xlimits.append(lims)

#Build up sromsize-to-SROM object map for plotting routine
sroms = OrderedDict()

for sromsize in sromsizes:

    #Generate SROM from file:
    srom = SROM(sromsize, target._dim)
    sromfile = "srom_m" + str(sromsize) + ".txt"
    sromfile = os.path.join(srom_dir, sromfile)
    srom.load_params(sromfile)
    sroms[sromsize] = srom
 
Postprocessor.compare_srom_CDFs(sroms, target, plotdir="plots",
                                plotsuffix=plot_suffix, variablenames=varz,                                     xlimits=xlimits, xticks=xticks,
                                cdfylabel=cdfylabel)

