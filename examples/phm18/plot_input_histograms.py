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

import numpy as np
import scipy
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.stats as st


title_font = {'fontname': 'Arial', 'size': 22, 'weight': 'bold',
                            'verticalalignment': 'bottom'}
axis_font = {'fontname': 'Arial', 'size': 30, 'weight': 'normal'}
lines = ['--', '-', ':', '-.']
colors = ['r', 'g', 'b', 'k']

label_font = 'Arial'
label_size = 26
legend_font = 16.5
line_width = 8

sample_file = "mc_data/input_samples_MC.txt"
samples = np.genfromtxt(sample_file)
n_samples = samples[:, 2]
y_samples = samples[:, 0]
c_samples = samples[:, 1]

n_limits = [np.min(n_samples), np.max(n_samples)]
y_limits = [np.min(y_samples), np.max(y_samples)]
c_limits = [np.min(c_samples), np.max(c_samples)]


n_bins = 20
y_bins = 20
c_bins = 20

# Plot n:

fig = plt.figure()
ax = plt.subplot()

plt.hist(n_samples, color='cornflowerblue', ec='black')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname(label_font)
    label.set_fontsize(label_size)
plt.xlabel(r'$n$', **axis_font)
plt.ylabel("Frequency", **axis_font)
for label in ax.xaxis.get_ticklabels()[0::2]:
    label.set_visible(False)
plt.tight_layout()
plt.savefig("plots/n_hist.pdf")
plt.show()

# Plot y0:
fig = plt.figure()
ax = plt.subplot()

plt.hist(y_samples, color='cornflowerblue', ec='black')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname(label_font)
    label.set_fontsize(label_size)
plt.xlabel(r'$y_0$', **axis_font)
plt.ylabel("Frequency", **axis_font)
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.tight_layout()
plt.savefig("plots/y_hist.pdf")
plt.show()


# Plot C
fig = plt.figure()
ax = plt.subplot()

plt.hist(c_samples, color='cornflowerblue', ec='black')

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname(label_font)
    label.set_fontsize(label_size)
plt.xlabel(r'log$C$', **axis_font)
plt.ylabel("Frequency", **axis_font)
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
plt.tight_layout()
plt.savefig("plots/c_hist.pdf")
plt.show()
