SROMPy - **S**tochastic **R**educed **O**rder **M**odels with **Py**thon 
==========================================================================

<a href='https://travis-ci.com/nasa/SROMPy'><img src='https://travis-ci.com/nasa/SROMPy.svg?branch=master' alt='Coverage Status' /></a> <a href='https://coveralls.io/github/lukemorrill/SROMPy?branch=master'><img src='https://coveralls.io/repos/github/lukemorrill/SROMPy/badge.svg?branch=master' alt='Coverage Status' /></a>

General
--------

Python module for generating Stochastic Reduced Order Models (SROMs) and applying them for uncertainty quantification problems. See documentation in `docs/` directory for details. 

Dependencies
-------------
SROMPy is intended for use with Python 2.7 and relies on the following packages:
* numpy
* scipy
* matplotlib
* mpi4py (optional for running in parallel)
* pytest (optional if the testing suite is to be run)

A requirements.txt file is included for easy installation of dependecies with pip:

```
pip install -r requirements.txt
```

Example Usage
--------------

```python
from SROMPy.postprocess import Postprocessor
from SROMPy.srom import SROM
from SROMPy.target import NormalRandomVariable

#Initialize Normal random variable object to be modeled by SROM:
normal = NormalRandomVariable(mean=3., std_dev=1.5)

#Initialize SROM & optimize to model the normal random variable:
srom = SROM(size=10, dim=1)
srom.optimize(normal)

#Compare the CDF of the SROM & target normal variable:
post_processor = Postprocessor(srom, normal)
post_processor.compare_CDFs()
```
  
The above code snippet produces the following CDF comparison plot: 
  
![CDF comparison](https://github.com/nasa/SROMPy/blob/master/examples/basic_tests/normal_rv_srom.png)

Getting Started
----------------
SROMPy can be installed via pip from [PyPI](https://pypi.org/project/SROMPy/):

```
pip install srompy
```

SROMPy can also be installed using the `git clone` command:

```
git clone https://github.com/nasa/SROMPy.git
```

The best way to get started with SROMPy is to take a look at the scripts in the examples/ directory. A simple example of propagating uncertainty through a spring mass system can be found in the examples/spring_mass/, while the examples/phm18/ directory contains scripts necessary to reproduce the results in the following conference paper on probabilistic prognostics: https://www.phmpapers.org/index.php/phmconf/article/view/551. For more information, see the source code documentation in docs/SROMPy_doc.pdf (a work in progress) or the technical report below that accompanied the release of SROMPy.

Tests
------
The tests can be performed by running "py.test" from the tests/ directory to ensure a proper installation.

Reference
-------------
If you use SROMPy for your research, please cite the technical report:

Warner, J. E. (2018). Stochastic reduced order models with Python (SROMPy). NASA/TM-2018-219824. 

The report can be found in the `docs/references` directory. Thanks!

Developers
-----------

UQ Center of Excellence <br />
NASA Langley Research Center <br /> 
Hampton, Virginia <br /> 

This software was funded by and developed under the High Performance Computing Incubator (HPCI) at NASA Langley Research Center. <br /> 

Contributors: James Warner (james.e.warner@nasa.gov), Luke Morrill, Juan Barrientos

License
---------

Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
 
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



