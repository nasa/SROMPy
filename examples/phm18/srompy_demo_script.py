import numpy
from postprocess import Postprocessor
from srom import SROM, SROMSurrogate
from target import SampleRV

#Define target random vector from samples
samplesfile = "input_samples_MC.txt"
MCsamples = numpy.genfromtxt(samplesfile)
target = SampleRV(MCsamples)

#Define SROM and determine optimal parameters
srom_size = 20
input_srom = SROM(size=srom_size, dim=3)
input_srom.optimize(target)

#Compare the input CDFs (produces Figure 3)
pp = Postprocessor(input_srom, target)
pp.compare_CDFs(variablenames=
                      [r'log$C$', r'$y_{0}$', r'$n$'])

#Run the model for each input SROM sample:
srom_eols = numpy.zeros(srom_size)
(srom_samples, srom_probs)=input_srom.get_params()
for i, sample in enumerate(srom_samples):
    srom_eols[i] = model.evaluate(sample)

#Generate SROM surrogate for the EOL
eol_srom = SROMSurrogate(input_srom, srom_eols)

#Make random variable with MC EOL solution
mc_eol_file = "eol_samples_MC.txt"
MC_eols = numpy.genfromtxt(mc_eol_file)
eol_mc = SampleRV(MC_eols)

#Compare final EOL solutions SROM vs MC:
# (produces Figure 7)
pp = Postprocessor(eol_srom, eol_mc)
pp.compare_CDFs(variablenames=["EOL"])


