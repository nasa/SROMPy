
import numpy as np
import scipy.optimize as opt
import scipy.stats as st

from target import SampleRV
from srom import SROM
from srom import Optimizer
from postprocess import Postprocessor

#--------------------------------------------------------------------

sromsize = 20
dim = 2
num_iterations = 5
joint_opt = False

#Generate samples of beta random variable as target rv being modeled by srom
a1 = 2.5
b1 = 0.5
a2 = 2.0
b2 = 5.0
numsamples = 2000

samples1 = st.beta.rvs(a1, b1, size=numsamples)
samples2 = st.beta.rvs(a2, b2, size=numsamples)
samples = np.array([samples1, samples2]).T

#Define target RV, SROM & obj. fun
target = SampleRV(samples)
srom_opt = Optimizer(target, sromsize)
srom = srom_opt.generate_srom(joint_opt, num_iterations)


pp = Postprocessor(srom, target)

pp.compare_CDFs()

