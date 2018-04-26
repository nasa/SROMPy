SROMPy - **S**tochastic **R**educed **O**rder **M**odels with **Py**thon 
==========================================================================

Python module for generating Stochastic Reduced Order Models (SROMs) and applying them for uncertainty quantification problems. See documentation in docs/ directory for details. 

Example usage:

```python
from postprocess import Postprocessor
from srom import SROM
from target import NormalRandomVariable

#Initialize Normal random variable object to be modeled by SROM:
normal = NormalRandomVariable(mean=3., std_dev=1.5)

#Initialize SROM & optimize to model the normal random variable:
srom = SROM(size=10, dim=1)
srom.optimize(normal)

#Compare the CDF of the SROM & target normal variable:
pp = Postprocessor(srom, normal)
pp.compare_CDFs()
```
  
The above code snippet produces the following CDF comparison plot: 
  
![CDF comparison](https://github.com/nasa/SROMPy/blob/master/examples/basic_tests/normal_rv_srom.png)

-------------------------------------------------------------------------------
If you use SROMPy for your research, please cite the technical report:

Warner, J. E. (2018). Stochastic reduced order models with Python (SROMPy). NASA/TM-2018-219824. 

The report can be found in the docs/references directory. Thanks!

-------------------------------------------------------------------------------

**Author**: <br />
James Warner <br />
UQ Center of Excellence <br />
NASA Langley Research Center <br /> 
james.e.warner@nasa.gov

-------------------------------------------------------------------------------

Notices:
Copyright 2018 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS." 

 Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
 
