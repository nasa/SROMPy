import numpy as np
import time
import pickle
from mpi4py import MPI
import sys

# TODO: FIND ALTERNATIVE TO THIS PYTHONPATH HACK
import os
import sys

# PYTHONPATH is not found when running mpiexec, so inject it so that we can
# load SROMPy modules...
if 'PYTHONPATH' not in os.environ:

    base_path = os.path.abspath('.')

    sys.path.insert(0, base_path)
    sys.path.insert(0, os.path.join(base_path, 'SROMPy'))

from SROMPy.srom import SROM
from SROMPy.target import BetaRandomVariable

# Get number of test samples from command line, or use default.
num_test_samples = 60  # LCM max # CPUs or a multiple of that is recommended.
if len(sys.argv) > 1:
    try:
        num_test_samples = int(sys.argv[1])
    except ValueError:
        print 'Argument must be a positive integer.'
        exit()

# Get MPI information.
comm = MPI.COMM_WORLD

if comm.rank == 0:
    print 'Will compute using %s samples on %s CPU%s.' % (num_test_samples, comm.size, 's' if comm.size > 1 else '')

# Load previously saved scalability data if available.
data_filename = os.path.join("data", "strong_scalability_data.txt")

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

# Random variable to optimize to.
random_variable = BetaRandomVariable(alpha=3., beta=2., shift=1., scale=2.5)

# performance data is two nested dictionaries.
# top level key is number of samples
# inner level has cpu number as keys with time data as values.
#
# ex: { 60: { 2: 5.2, 3: 1.8 } }
#
# In above example have two tests both with 60 samples;
# one with 2 cpus and one with 3 cpus.

if num_test_samples not in performance_data:
    performance_data[num_test_samples] = {}

# Repeat performance data collection in order to smooth the chart lines.
iteration_performance = []
for i in xrange(10):

    input_srom = SROM(20, 1)

    t0 = time.time()
    input_srom.optimize(random_variable, num_test_samples=num_test_samples)
    iteration_performance.append(time.time() - t0)

mean_performance = np.mean(iteration_performance)

if comm.rank != 0:
    exit()

performance_data[num_test_samples][comm.size] = mean_performance

print '%s test samples across %s CPUs took %s seconds.' % (num_test_samples, comm.size, float(mean_performance))

print 'Writing results to disk...'
print performance_data

if not os.path.isdir('data'):
    os.mkdir('data')

data_file = open(data_filename, 'w')
pickle.dump(performance_data, data_file)
data_file.close()
