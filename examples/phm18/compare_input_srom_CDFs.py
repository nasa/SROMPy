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

# Target Monte Carlo input samples for comparison.
target_samples = "mc_data/input_samples_MC.txt"

# SElect 3 SROM sizes.
srom_sizes = [5, 10, 20]
srom_dir = "srom_data"

# Plotting specs:
variables = [r'$y_{0}$', r'log$C$', r'$n$']
cdf_y_label = True        # Label y axis as "CDF"
plot_dir = "plots"
plot_suffix = "SROM_input_CDF_m"
for m in srom_sizes:
    plot_suffix += "_" + str(m)

# x_tick labels for each variable for clarity.
y0_ticks = ['', '0.245', '', '0.255', '', '0.265', '', '0.275']
log_c_ticks = ['', '-8.8', '', '-8.4', '', '-8.0', '', '-7.6']
n_ticks = ['1.0', '', '1.5', '', '2.0', '', '2.5', '', '3.0']
x_ticks = [y0_ticks, log_c_ticks, n_ticks]

# Load / initialize target random variable from samples:
samples = np.genfromtxt(target_samples)
target = SampleRandomVector(samples)

# Set x limits for each variable based on target:
x_limits = []
for i in range(target._dim):
    limits = [np.min(samples[:, i]), np.max(samples[:, i])]
    x_limits.append(limits)

# Build up srom_size-to-SROM object map for plotting routine.
sroms = OrderedDict()

for srom_size in srom_sizes:

    # Generate SROM from file:
    srom = SROM(srom_size, target._dim)
    srom_filename = "srom_m" + str(srom_size) + ".txt"
    srom_filename = os.path.join(srom_dir, srom_filename)
    srom.load_params(srom_filename)
    sroms[srom_size] = srom
 
# Font size specs & plotting.
axis_font_size = 25
legend_font_size = 20
Postprocessor.compare_srom_cdfs(sroms, target, plot_dir="plots",
                                plot_suffix=plot_suffix,
                                variable_names=variables,
                                x_limits=x_limits,
                                x_ticks=x_ticks,
                                cdf_y_label=cdf_y_label,
                                axis_font_size=axis_font_size,
                                legend_font_size=legend_font_size)

