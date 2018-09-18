from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import NormalRandomVariable

#Initialize Normal random variable object to be modeled by SROM:
normal = NormalRandomVariable(mean=3., std_dev=1.5)

#Initialize SROM & optimize to model the normal random variable:
srom = SROM(size=10, dim=1)
srom.optimize(normal)

#Compare the CDF of the SROM & target normal variable:
pp = Postprocessor(srom, normal)
pp.compare_CDFs()

