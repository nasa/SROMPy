import os
import numpy as np
from collections import OrderedDict

from target import SampleRV
from srom import SROM, SROMSurrogate
from postprocess import Postprocessor

'''
Compare piecewise constant SROM approximations to the EOL for m=5,10,20
Produces Figure 5(a) in the paper
'''

#Target Monte Carlo input samples for comparison
targetsamples = "mc_data/eol_samples_MC.txt"

#SElect 3 SROM sizes
sromsizes = [5,10,20]
srom_dir = "srom_data"

#Plotting specs:
varz = [r'EOL (Cycles)']
xlimits = [[1.0e6, 2.0e6]]
ylimits = [[-0.01, 1.1]]
xticks = [[r'$1.0 \times 10^6$','',r'$1.4 \times 10^6$','',
           r'$1.8 \times 10^6$','']]
xaxispadding = 5
axisfontsize = 28
labelfontsize = 24
legendfontsize = 24
showplot = False
cdfylabel = True        #Label y axis as "CDF"
plot_dir = "plots"
plot_suffix = "SROM_pwconst_eol_CDF_m"
for m in sromsizes:
    plot_suffix += "_" + str(m)

#Load / initialize target random variable from samples:
samples = np.genfromtxt(targetsamples)
target = SampleRV(samples)

#Set x limits for each variable based on target:
#xlimits = [[np.min(samples), np.max(samples)]]

#Build up sromsize-to-SROM object map for plotting routine
sroms = OrderedDict()

for sromsize in sromsizes:

    #Generate input SROM from file:
    srom = SROM(sromsize, target._dim)
    sromfile = "srom_m" + str(sromsize) + ".txt"
    sromfile = os.path.join(srom_dir, sromfile)
    srom.load_params(sromfile)
        
    #Generate SROM surrogate for output from EOLs & input srom:
    eolfile = "srom_eol_m" + str(sromsize) + ".txt"
    eolfile = os.path.join(srom_dir, eolfile)
    eols = np.genfromtxt(eolfile)

    sroms[sromsize] = SROMSurrogate(srom, eols)
 
Postprocessor.compare_srom_CDFs(sroms, target, plotdir="plots",
                                plotsuffix=plot_suffix, variablenames=varz,                                     xlimits=xlimits, ylimits=ylimits, xticks=xticks,
                                cdfylabel=True, xaxispadding=xaxispadding,
                                axisfontsize=axisfontsize, 
                                labelfontsize=labelfontsize,
                                legendfontsize=legendfontsize)

