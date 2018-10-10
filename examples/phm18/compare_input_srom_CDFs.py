# Copyright 2018 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.

# The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed
# under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import os
import numpy as np
from collections import OrderedDict

from SROMPy.target import SampleRandomVector
from SROMPy.srom import SROM
from SROMPy.postprocess import Postprocessor

'''
Compare SROMs for inputs - produce figure 3 in the paper
'''

#Target Monte Carlo input samples for comparison
targetsamples = "mc_data/input_samples_MC.txt"

#SElect 3 SROM sizes
sromsizes = [5,10,20]
srom_dir = "srom_data"

#Plotting specs:
varz = [r'$y_{0}$', r'log$C$', r'$n$']
cdfylabel = True        #Label y axis as "CDF"
plot_dir = "plots"
plot_suffix = "SROM_input_CDF_m"
for m in sromsizes:
    plot_suffix += "_" + str(m)

#Xtick labels for each variable for clarity
y0ticks = ['', '0.245', '', '0.255', '', '0.265', '', '0.275']
logCticks = ['', '-8.8', '', '-8.4', '', '-8.0', '', '-7.6']
nticks = ['1.0', '', '1.5', '', '2.0', '', '2.5', '', '3.0']
xticks = [y0ticks, logCticks, nticks]

#Load / initialize target random variable from samples:
samples = np.genfromtxt(targetsamples)
target = SampleRandomVector(samples)

#Set x limits for each variable based on target:
xlimits = []
for i in range(target.dim):
    lims = [np.min(samples[:,i]), np.max(samples[:,i])]
    xlimits.append(lims)

#Build up sromsize-to-SROM object map for plotting routine
sroms = OrderedDict()

for sromsize in sromsizes:

    #Generate SROM from file:
    srom = SROM(sromsize, target.dim)
    sromfile = "srom_m" + str(sromsize) + ".txt"
    sromfile = os.path.join(srom_dir, sromfile)
    srom.load_params(sromfile)
    sroms[sromsize] = srom
 
#Font size specs & plotting
axisfontsize = 25
legendfontsize = 20
Postprocessor.compare_srom_CDFs(sroms, target, plotdir="plots",
                                plotsuffix=plot_suffix, variablenames=varz,                                     xlimits=xlimits, xticks=xticks,
                                cdfylabel=cdfylabel, axisfontsize=axisfontsize,
                                legendfontsize=legendfontsize)

