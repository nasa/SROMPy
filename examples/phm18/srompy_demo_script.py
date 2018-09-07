import numpy
from os import path
from src.postprocess import Postprocessor
from src.srom import SROM, SROMSurrogate
from src.target import SampleRandomVector

#Define target random vector from samples
monte_carlo_input_samples_filename = path.join("mc_data", "input_samples_MC.txt")
monte_carlo_input_samples = numpy.genfromtxt(monte_carlo_input_samples_filename)
target_vector = SampleRandomVector(monte_carlo_input_samples)

#Define SROM and determine optimal parameters
srom_size = 20
input_srom = SROM(size=srom_size, dim=3)
input_srom.optimize(target_vector)

#Compare the input CDFs (produces Figure 6)
post_processor = Postprocessor(input_srom, target_vector)
post_processor.compare_CDFs(variablenames=
                            [r'$y_{0}$', r'log$C$', r'$n$'])

#Run the model for each input SROM sample:
srom_results = numpy.zeros(srom_size)
(srom_samples, srom_probs) = input_srom.get_params()

# TODO: define model here.
model = None

if model is None:
    raise ValueError("model has not been defined.")

for i, sample in enumerate(srom_samples):
    srom_results[i] = model.evaluate(sample)

#Generate SROM surrogate for the end of life
srom_surrogate_model = SROMSurrogate(input_srom, srom_results)

#Make random variable with MC end of life solution
monte_carlo_results_filename = "eol_samples_MC.txt"
monte_carlo_results_samples = numpy.genfromtxt(monte_carlo_results_filename)
target_vector = SampleRandomVector(monte_carlo_results_samples)

#Compare final EOL solutions SROM vs MC:
# (produces Figure 7)
post_processor = Postprocessor(srom_surrogate_model, target_vector)
post_processor.compare_CDFs(variablenames=["EOL"])
