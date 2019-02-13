
Example - Spring Mass System
=============================

This example will use SROMPy to simulate a spring-mass system with random spring stiffness (:ref:`spring-mass`). The example covers modeling the random stiffness using a Beta random variable in SROMPy, generating a SROM to represent the stiffness, then propagating uncertainty though the model to obtain the distribution for maximum displacement. The SROM solution will be compared to standard Monte Carlo simulation.

.. _spring-mass:

.. figure:: images/spring_mass_diagram.png
    :align: center
    :width: 2in

    Spring-mass system

The governing equation of motion for the system is given by

.. math:: m_s \ddot{z}  = -k_s z + m_s g
    :label: springmass

where :math:`m_s` is the mass, :math:`k_s` is the spring stiffness, :math:`g` 
is the acceleration due to gravity, :math:`z` is the vertical displacement 
of the mass, and :math:`\ddot{z}` is the acceleration of the mass. The 
source of uncertainty in the system will be the spring stiffness, which is 
modeled as a random variable of the following form:

.. math:: K_s = \gamma + \eta B 
    :label: random-stiffness

where :math:`\gamma` and :math:`\eta` are shift and scale parameters, 
respectively, and :math:`B = \text{Beta}(\alpha, \beta)` is a standard Beta 
random variable with shape parameters :math:`\alpha` and :math:`\beta`. Let 
these parameters take the following values: :math:`\gamma=1.0N/m`, 
:math:`\eta = 2.5N/m`, :math:`\alpha=3.0`, and :math:`\beta=2.0`. The mass 
is assumed to be deterministic, :math:`m_s = 1.5kg`, and the acceleration due 
to gravity is :math:`g = 9.8 m^2/s`. 


With uncertainty in an input parameter, the resulting displacement, :math:`Z`, is a random variable as well. The quantity of interest in this example with be the maximum displacement over a specified time window, :math:`Z_{max}=max_t(Z)`. It is assumed we have access to a computational model that numerically integrates the governing equation over this time window for a given sample of the random stiffness and returns the maximum displacement. The goal of this example will be to approximate the CDF, :math:`F(z_{max})`, using the SROM approach with SROMPy and compare it to a Monte Carlo simulation solution.


Step 1: Define target random variable, initialize model, generate reference solution
-------------------------------------------------------------------------------------
Begin by importing the needed SROMPy classes as well as the SpringMass1D class that defines the spring mass model:

.. code-block:: python

  import numpy as np

  #import SROMPy modules
  from model import SpringMass_1D
  from postprocess import Postprocessor
  from srom import SROM, SROMSurrogate, FiniteDifference as FD
  from target import SampleRV, BetaRandomVariable 

The first step in the analysis is to define the target random variable to represent the spring stiffness :math:`K_s` using the BetaRandomVariable class in SROMPy:

.. code-block:: python

  #Random variable for spring stiffness
  stiffness_rv = BetaRandomVariable(alpha=3.,beta=2.,shift=1.,scale=2.5)

Next, the computational model of the spring-mass system is initialized:

.. code-block:: python
    
  #Specify spring-mass system and initialize model:
  m = 1.5                           #deterministic mass
  state0 = [0., 0.]                 #initial conditions at rest
  t_grid = np.arange(0., 10., 0.1)  #time discretization
  model = SpringMass_1D(m, state0, t_grid)

A reference solution using Monte Carlo simulation is now generated for comparison later on. This is done by sampling the random spring stiffness, evaluating the model for each sample, and then using the SROMPy SampleRV class to represent the Monte Carlo solution for maximum displacement:

.. code-block:: python

  #----------Monte Carlo------------------
  #Generate stiffness input samples for Monte Carlo
  num_samples = 5000
  stiffness_samples = stiffness_rv.draw_random_sample(num_samples)

  #Calculate maximum displacement samples using MC simulation
  disp_samples = np.zeros(num_samples)
  for i, stiff in enumerate(stiffness_samples):
      disp_samples[i] = model.get_max_disp(stiff)

  #Get Monte carlo solution as a sample-based random variable:
  mc_solution = SampleRV(disp_samples)


Step 2: Construct SROM for the input
-------------------------------------

A SROM, :math:`\tilde{K}_s` is now formed to model the random stiffness input, :math:`K_s`, with SROMPy. The following code initializes the SROM class for a model size of 10 and uses the optimize function to set the optimal SROM parameters to represent the random spring stiffness:

.. code-block:: python
    
  #Generate SROM for random stiffness
  sromsize = 10
  dim = 1
  input_srom = SROM(sromsize, dim)
  input_srom.optimize(stiffness_rv)

The CDF of the resulting SROM can be compared to the original Beta random variable for spring stiffness using the SROMPy Postprocessor class:

.. code-block:: python

  #Compare SROM vs target stiffness distribution:
  pp_input = Postprocessor(input_srom, stiffness_rv)
  pp_input.compare_CDFs()

This produces the following plot:

.. _input-srom:

.. figure:: images/stiffness_CDFs.png
    :align: center
    :width: 4in

Step 3: Evaluate model for each SROM sample:
---------------------------------------------
Now output samples of maximum displacement must be generated by running the spring-mass model for each stiffness sample from the input SROM, i.e., 

:math:`\tilde{z}^{(k)}_{max} = \mathcal{M}(\tilde{k}_s^{(k)}) \; \text{for } \; k=1,...,m`

This is done with the following code:

.. code-block:: python

  #run model to get max disp for each SROM stiffness sample
  srom_disps = np.zeros(sromsize)
  (samples, probs) = input_srom.get_params()
  for i, stiff in enumerate(samples):
      srom_disps[i] = model.get_max_disp(stiff)



Step 4: Form SROM surrogate model for output
----------------------------------------------

Approach a) Piecewise-constant approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple piecewise-constant approximation to the output (maximum displacement) can be generated with the SROMSurrogate class using the input SROM formed previously and the calculated maximum displacement samples:


.. code-block:: python

  #Form SROM surrogate for the max disp. solution using samples from the model and input SROM:
  output_srom = SROMSurrogate(input_srom, srom_disps)

Compare the SROM approximation to the maximum displacement CDF against the Monte Carlo solution:

.. code-block:: python

  #Compare solutions
  pp_output = Postprocessor(output_srom, mc_solution)
  pp_output.compare_CDFs(variablenames=[r'$Z_{max}$'])

This produces the following comparison plot:

.. _output-pwc-srom:

.. figure:: images/disp_CDFs_pw_constant.png
    :align: center
    :width: 4in


Approach b) Piecewise-linear approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now a more accurate piecewise-linear SROM surrogate model is formed to estimate the CDF of the maximum displacement. To do so, gradients must be calculated using finite difference and provided to the SROMSurrogate class upon initialization.

The finite different gradients are calculated with the help of the FiniteDifference class (FD), requiring extra model evaluations for perturbed inputs:


.. code-block:: python

  #Perturbation size for finite difference
  stepsize = 1e-12
  samples_fd = FD.get_perturbed_samples(samples, perturb_vals=[stepsize])

  #Run model to get perturbed outputs for FD calc.
  perturbed_disps = np.zeros(sromsize)
  for i, stiff in enumerate(samples_fd):
      perturbed_disps[i] = model.get_max_disp(stiff)
  gradient = FD.compute_gradient(srom_disps, perturbed_disps, [stepsize])


A piecewise-linear surrogate model can now be constructed and then sampled to approximate the CDF of the maximum displacement:

.. code-block:: python

  #Initialize piecewise-linear SROM surrogate w/ gradients:
  surrogate_PWL = SROMSurrogate(input_srom, srom_disps, gradient)

  #Use the surrogate to produce max disp samples from the input stiffness samples:
  output_samples = surrogate_PWL.sample(stiffness_samples)

  #Represent the SROM solution as a sample-based random variable:
  solution_PWL = SampleRV(output_samples)

Finally, the new piece-wise linear CDF approximation is compared to the Monte Carlo solution:

.. code-block:: python

  #Compare SROM piecewise linear solution to Monte Carlo
  pp_pwl = Postprocessor(solution_PWL, mc_solution)
  pp_pwl.compare_CDFs(variablenames=[r'$Z_{max}$'])


.. _output-pwl-srom:

.. figure:: images/disp_CDFs_pw_linear.png
    :align: center
    :width: 4in
