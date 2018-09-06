import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load data. Should be a file with performance data for each number of CPUs.
all_speedup_data = []

print 'Loading performance data.'

data_filename = os.path.join("data", "weak_scalability_data.txt")
if os.path.isfile(data_filename):

    data_file = open(data_filename, 'r')
    performance_data = pickle.load(data_file)
    serial_performance = np.array(performance_data[1])
    data_file.close()

    min_samples = performance_data['min_samples']
    max_samples = performance_data['max_samples']
    samples_step = performance_data['samples_step']

else:

    print 'Chart data not found!'
    exit()

# Get parallel performance data for as many data files as are available.
num_cpus = 2
while num_cpus in performance_data:

    parallel_performance = np.array(performance_data[num_cpus])
    all_speedup_data.append(serial_performance / parallel_performance)

    num_cpus += 1

print 'Found data for up to %s CPUs.' % str(num_cpus - 1)

# Generate the plots.
fig, ax = plt.subplots(1)

x_values = np.arange(min_samples, max_samples, samples_step)

for i, speedup_data in enumerate(all_speedup_data):

    label_text = str(i + 2) + ' CPUs'
    ax.plot(x_values, speedup_data, label=label_text)

# Add labels, legend, and title.
plt.xlabel('Number of Samples')
plt.ylabel('Speedup factor')
plt.legend()
plt.ylim(0, 10)  # May need adjustment to fit legend without overlapping data.

fig.canvas.set_window_title("Weak Scalability")
plt.title("Parallel Speedup Factor vs Number of Samples")

plt.show()
