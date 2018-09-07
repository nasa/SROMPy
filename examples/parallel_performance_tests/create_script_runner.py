import os

part_one = 'mpiexec -n '
part_two = ' python generate_data_strong_scalability.py '

script_filename = 'generate_data.sh'
script_file = open(script_filename, 'w')

for num_cpus in range(1, 7):
    for num_samples in [60, 180, 320, 640]:

        command = part_one + str(num_cpus) + part_two + str(num_samples) + '\n'
        script_file.write(command)

script_file.close()
print 'Done!'
