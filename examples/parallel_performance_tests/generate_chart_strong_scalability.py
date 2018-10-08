import matplotlib.pyplot as plt
import pickle
import os

# Load performance data, which is a pickled dictionary.
data_filename = os.path.join("data", "strong_scalability_data.txt")

all_performance_data = {}
if os.path.isfile(data_filename):

    print 'Loading previously saved data set for modification:'

    data_file = open(data_filename, 'r')
    all_performance_data = pickle.load(data_file)
    data_file.close()

    print all_performance_data

else:

    print 'Chart data not found!'
    exit()

# Generate the plots.
fig, ax = plt.subplots(1)

# Generate a line in the chart for each sample size.
for num_samples in all_performance_data.keys():

    performance_data = all_performance_data[num_samples]
    baseline_time = performance_data[1]

    x_values = []
    actual_y_values = []
    cpu_num = 1

    while cpu_num in performance_data:

        x_values.append(cpu_num)
        actual_y_values.append(baseline_time / performance_data[cpu_num])

        cpu_num += 1

    plt.plot(x_values, actual_y_values, label=str(num_samples) + ' samples')

plt.plot(x_values, x_values, 'r--', label='Ideal', linewidth=2)

# Add labels, legend, and title.
plt.xlabel('Number of CPUs')
plt.ylabel('Speedup factor')
plt.legend()

fig.canvas.set_window_title("Strong Scalability ")
plt.title("Speedup vs Number of CPUs")

plt.show()
