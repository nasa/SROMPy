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
from SROMPy.srom import SROM, SROMSurrogate
from SROMPy.postprocess import Postprocessor

'''
Compare piecewise constant SROM approximations to the EOL for m=5,10,20
Produces Figure 5(a) in the paper
'''

# Target Monte Carlo input samples for comparison.
target_samples = "mc_data/eol_samples_MC.txt"

# SElect 3 SROM sizes.
srom_sizes = [5, 10, 20]
srom_dir = "srom_data"

# Plotting specs:
variables = [r'EOL (Cycles)']
x_limits = [[1.0e6, 2.0e6]]
y_limits = [[-0.01, 1.1]]
x_ticks = [[r'$1.0 \times 10^6$', '', r'$1.4 \times 10^6$', '',
           r'$1.8 \times 10^6$','']]
x_axis_padding = 5
axis_font_size = 24
label_font_size = 20
legend_font_size = 20
show_plot = False
cdf_y_label = True        # Label y axis as "CDF".
plot_dir = "plots"
plot_suffix = "SROM_pwconst_eol_CDF_m"
for m in srom_sizes:
    plot_suffix += "_" + str(m)

# Load / initialize target random variable from samples:
samples = np.genfromtxt(target_samples)
target = SampleRandomVector(samples)

# Build up srom_size-to-SROM object map for plotting routine
sroms = OrderedDict()

for srom_size in srom_sizes:

    # Generate input SROM from file:
    srom = SROM(srom_size, target._dim)
    srom_filename = "srom_m" + str(srom_size) + ".txt"
    srom_filename = os.path.join(srom_dir, srom_filename)
    srom.load_params(srom_filename)
        
    # Generate SROM surrogate for output from EOLs & input srom:
    end_of_life_filename = "srom_eol_m" + str(srom_size) + ".txt"
    end_of_life_filename = os.path.join(srom_dir, end_of_life_filename)
    end_of_life_data = np.genfromtxt(end_of_life_filename)

    sroms[srom_size] = SROMSurrogate(srom, end_of_life_data)
 
Postprocessor.compare_srom_cdfs(sroms, target, plot_dir="plots",
                                plot_suffix=plot_suffix,
                                variable_names=variables, y_limits=y_limits,
                                x_ticks=x_ticks, cdf_y_label=True,
                                x_axis_padding=x_axis_padding,
                                axis_font_size=axis_font_size,
                                label_font_size=label_font_size,
                                legend_font_size=legend_font_size)

