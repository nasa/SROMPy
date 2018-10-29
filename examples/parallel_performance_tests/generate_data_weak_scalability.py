import numpy as np
import time
import pickle
from mpi4py import MPI

# TODO: FIND ALTERNATIVE TO THIS PYTHONPATH HACK
import os
import sys

# PYTHONPATH is not found when running mpiexec,
# so inject it so that we can load SROMPy modules...
if 'PYTHONPATH' not in os.environ:
    base_path = os.path.abspath('.')
    sys.path.insert(0, base_path)
    sys.path.insert(0, os.path.join(base_path, 'SROMPy'))

from SROMPy.srom import SROM
from SROMPy.target import BetaRandomVariable

# Random variable to optimize to.
random_variable = BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

# Run each SROM optimization 10 times to generate performance data.
performance_data = []

# Get MPI information.
comm = MPI.COMM_WORLD

# Load previously saved scalability data if available.
data_filename = os.path.join("data", "weak_scalability_data.txt")

if os.path.isfile(data_filename):

    if comm.rank == 0:
        print 'Loading previously saved data set for modification:'

    data_file = open(data_filename, 'r')
    performance_data = pickle.load(data_file)
    data_file.close()

    if comm.rank == 0:
        print performance_data

else:

    if comm.rank == 0:
        print 'No previous performance data found. Creating new data set.'

    performance_data = {}

if comm.rank == 0:
    print 'Performing computation with %s CPUs.' % comm.size

min_samples = 6
max_samples = 301
samples_step = 30

performance_data['min_samples'] = min_samples
performance_data['max_samples'] = max_samples
performance_data['samples_step'] = samples_step

test_results = []
for num_test_samples in xrange(min_samples, max_samples, samples_step):

    if comm.rank == 0:
        print 'Optimizing with %s test samples...' % num_test_samples

    # Repeat performance data collection in order to smooth the chart lines.
    iteration_performance = []
    for i in xrange(10):

        input_srom = SROM(20, 1)

        t0 = time.time()
        input_srom.optimize(random_variable, num_test_samples=num_test_samples)
        iteration_performance.append(time.time() - t0)

    # We only need to save performance data from one process because
    # they should be the same due to gather operation within optimize().
    if comm.rank == 0:

        mean_performance = np.mean(iteration_performance)
        test_results.append(mean_performance)

if comm.rank != 0:
    exit()

performance_data[comm.size] = test_results

# Create data directory to store output if necessary.
if not os.path.isdir('data'):
    os.mkdir('data')

data_file = open(data_filename, 'w')
pickle.dump(performance_data, data_file)
data_file.close()
