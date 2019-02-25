SROMPy - **S**tochastic **R**educed **O**rder **M**odels with **Py**thon 
==========================================================================

Python module for generating Stochastic Reduced Order Models (SROMs) and applying them for uncertainty quantification problems. See documentation in `docs/` directory for details. 


Example usage:

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

-------------------------------------------------------------------------------
If you use SROMPy for your research, please cite the technical report:

Warner, J. E. (2018). Stochastic reduced order models with Python (SROMPy). NASA/TM-2018-219824. 

The report can be found in the `docs/references` directory. Thanks!

-------------------------------------------------------------------------------

**Authors**: <br />
James Warner <br />
UQ Center of Excellence <br />
NASA Langley Research Center <br /> 
james.e.warner@nasa.gov

Luke Morrill <br />
Georgia Tech

This software was funded by and developed under the High Performance Computing Incubator (HPCI) at NASA Langley Research Center.

-------------------------------------------------------------------------------

Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 
 
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



