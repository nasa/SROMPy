'''
Class to solve SROM optimization problem.
'''

import numpy as np
import scipy.optimize as opt
import time
import imp

from src.optimize import ObjectiveFunction
from src.optimize import Gradient


#------------Helper funcs for scipy optimize-----------------------------
def scipy_obj_fun(x,
                  objfun,
                  grad,
                  samples):
    '''
    Function to pass to scipy minimize defining objective. Wraps the
    ObjectiveFunction.evaluate() function that defines SROM error. Need to
    unpack design variables x into samples & probabilities. Handle two cases:

    1) joint optimization over samples & probs (samples=None)
    2) sequential optimization -> optimize probs for fixed samples
    '''

    size = objfun._SROM._size
    dim = objfun._SROM._dim

    #Unpacking simple with samples are fixed:
    probs = x

    error = objfun.evaluate(samples, probs)

    return error

def scipy_grad(x,
               objfun,
               grad,
               samples):
    '''
    Function to pass to scipy minimize defining objective. Wraps the
    ObjectiveFunction.evaluate() function that defines SROM error. Need to
    unpack design variables x into samples & probabilities. Handle two cases:

    1) joint optimization over samples & probs (samples=None)
    2) sequential optimization -> optimize probs for fixed samples
    '''

    size = grad._SROM._size
    dim = grad._SROM._dim

    #Unpacking simple with samples are fixed:
    probs = x

    grad = grad.evaluate(samples, probs)
    return grad

#-----------------------------------------------------------------


class Optimizer:
    '''
    Class that delegates the construction of an SROM through the optimization
    of the SROM parameters (samples/probs) that minimize the error between
    SROM & target random vector
    '''

    def __init__(self,
                 target,
                 srom,
                 obj_weights=None,
                 error='SSE',
                 max_moment=5,
                 cdf_grid_pts=100):
        '''

        inputs:
            -target - initialized RandomVector object (either AnalyticRandomVector or
                SampleRandomVector)
            -obj_weights - array of floats defining the relative weight of the
                terms in the objective function. Terms are error in moments,
                CDFs, and correlation matrix in that order. Default will give
                each term equal weight
            -error - string 'mean' or 'max' defining how error is defined
                between the statistics of the SROM & target
            -max_moment - int, max order to evaluate moment errors up to
            -cdf_grid_pts - int, # pts to evaluate CDF errors on

        '''

        self._target = target

        # Initialize objective function defining SROM vs target error.
        self._srom_obj = ObjectiveFunction(srom,
                                           target,
                                           obj_weights,
                                           error,
                                           max_moment,
                                           cdf_grid_pts)

        self._srom_grad = Gradient(srom,
                                   target,
                                   obj_weights,
                                   error,
                                   max_moment,
                                   cdf_grid_pts)

        # Get srom size & dimension.
        self._srom_size = srom.get_size()
        self._dim = srom.get_dim()

        # Gradient only available for SSE error obj function.
        if error.upper() == "SSE":
            self._grad = scipy_grad
        else:
            self._grad = None

        # Test whether parallel processing is available.
        try:
            imp.find_module('mpi4py')

            import mpi4py
            comm = mpi4py.MPI.COMM_WORLD

            self.number_CPUs = comm.size
            self.cpu_rank = comm.rank

        except ImportError:

            self.number_CPUs = 1
            self.cpu_rank = 0

    def get_optimal_params(self,
                           num_test_samples=500,
                           tol=None,
                           options=None,
                           method=None,
                           joint_opt=False,
                           output_interval=10,
                           verbose=True):
        '''
        Solve the SROM optimization problem - finds samples & probabilities
        that minimize the error between SROM/Target RV statistics.

        inputs:
            -joint_opt, bool, Flag for optimizing jointly for samples & probs
                rather than sequentially (draw samples then optimize probs in
                loop - default).
            -num_test_samples, int, If optimizing sequentially (samples then
                probs), this is number of random sample sets to test in opt
            -tol, float, tolerance of scipy optimization algorithm
            -options, dict, options for scipy optimization algorithm
            -method, str, method specifying scipy optimization algorithm
            -output_interval, int, how often to print optimization progress

        returns optimal SROM samples & probabilities

        '''

        if not isinstance(num_test_samples, int):
            raise TypeError("Number number of test samples must be a positive integer.")

        if num_test_samples <= 0:
            raise ValueError("Insufficient number of test samples specified.")

        bounds = self.get_param_bounds(joint_opt, self._srom_size)
        constraints = self.get_constraints(joint_opt, self._srom_size, self._dim)
        initial_guess = self.get_initial_guess(joint_opt, self._srom_size)

        # Track optimal func value with corresponding samples/probs.
        optimal_probabilities = None
        optimal_samples = None
        best_objective_function_result = 1e6

        # Report whether we're running in sequential or parallel mode.
        if verbose:
            if self.number_CPUs == 1:
                print "SROM Sequential Optimizer:"
            elif self.cpu_rank == 0:
                print "SROM Parallel Optimizer (%s cpus):" % self.number_CPUs

            if verbose and self.cpu_rank == 0 and num_test_samples % self.number_CPUs != 0:
                print "Warning: specified number of test samples cannot be equally distributed among processors!"
                print "%s per core, %s total" % (num_test_samples // self.number_CPUs, num_test_samples)

        np.random.seed(self.cpu_rank)
        num_test_samples_per_cpu = num_test_samples // self.number_CPUs
        t0 = time.time()

        for i in xrange(num_test_samples_per_cpu):

            # Randomly draw new.
            srom_samples = self._target.draw_random_sample(self._srom_size)

            # Optimize using scipy.
            optimization_result = opt.minimize(scipy_obj_fun, initial_guess,
                                               args=(self._srom_obj, self._srom_grad,
                                                     srom_samples),
                                               jac=self._grad,
                                               constraints=constraints,
                                               method=method,
                                               bounds=bounds)

            # If error is lower than lowest so far, keep track of results.
            if optimization_result['fun'] < best_objective_function_result:
                optimal_samples = srom_samples
                optimal_probabilities = optimization_result['x']
                best_objective_function_result = optimization_result['fun']

            # Reporting ongoing results to user if in sequential mode (parallel has confusing output).
            if verbose and self.number_CPUs == 1 and (i == 0 or (i + 1) % output_interval == 0):
                print "\tIteration", i + 1, "Objective Function:", optimization_result['fun'],
                print "Optimal:", best_objective_function_result

        # If we're running in parallel mode, we need to gather all of the data and identify the best result.
        if self.number_CPUs > 1:

            if verbose and self.cpu_rank == 0:
                print 'Collecting and comparing results from each cpu...'

            # Create a package to transmit results in.
            this_cpu_results = {'samples': optimal_samples, 'probabilities': optimal_probabilities}

            # Gather results.
            import mpi4py
            comm = mpi4py.MPI.COMM_WORLD
            all_cpu_results = comm.gather(this_cpu_results, root=0)

            if self.cpu_rank == 0:

                best_mean_error = 1e6

                for result in all_cpu_results:

                    result_moment_error = self._srom_obj.get_moment_error(result['samples'], result['probabilities'])
                    result_cdf_error = self._srom_obj.get_cdf_error(result['samples'], result['probabilities'])
                    result_correlation_error = self._srom_obj.get_corr_error(result['samples'], result['probabilities'])

                    result_mean_error = np.mean([result_moment_error, result_cdf_error, result_correlation_error])

                    if result_mean_error < best_mean_error:
                        best_mean_error = result_mean_error
                        optimal_samples = result['samples']
                        optimal_probabilities = result['probabilities']

        # Display final errors in statistics:
        moment_error = self._srom_obj.get_moment_error(optimal_samples, optimal_probabilities)
        cdf_error = self._srom_obj.get_cdf_error(optimal_samples, optimal_probabilities)
        correlation_error = self._srom_obj.get_corr_error(optimal_samples, optimal_probabilities)

        if verbose and self.cpu_rank == 0:
            print "\tOptimization time: ", time.time() - t0, "seconds"
            print "\tFinal SROM errors:"
            print "\t\tCDF: ", cdf_error
            print "\t\tMoment: ", moment_error
            print "\t\tCorrelation: ", correlation_error

        return optimal_samples, optimal_probabilities

    #-----Helper funcs----
    
    def get_param_bounds(self, joint_opt, sromsize):
        '''
        Get the bounds on parameters for SROM optimization problem. If doing
        joint optimization, need bounds for both samples & probs. If not,
        just need trivial bounds on probabilities
        '''
        
        if not joint_opt:
            bounds = [(0.0,1.0)]*sromsize
        else:
            raise NotImplementedError("SROM joint optimization not implemented")

        return bounds
        
    def get_constraints(self, joint_opt, sromsize, dim):
        '''
        Returns constraint dictionaries for scipy optimize that enforce 
        probabilities summing to 1 for joint or sequential optimize case
        '''

        # A little funky, need to return function as constraint.
        #TODO - use lambda function instead?

        # Sequential case - unknown vector x is probabilities directly
        def seq_constraint(x):
            return 1.0 - np.sum(x)

        # Joint case - probabilities at end of unknown vector x
        def joint_constraint(x, sromsize, dim):
            return 1.0 - np.sum(x[sromsize*dim:])

        if not joint_opt:
            return {'type':'eq', 'fun':seq_constraint}
        else:
            return {'type':'eq', 'fun':joint_constraint, 'args':(sromsize, dim)}
        
        
    def get_initial_guess(self, joint_opt, sromsize):
        '''
        Return initial guess for optimization. Randomly drawn samples w/ equal
        probability for joint optimization or just equal probabilities for 
        sequential optimization
        '''
    
        if joint_opt:
            # Randomly draw some samples & hstack them with probabilities
            # TODO - untested
            samples = self._target.draw_random_sample(sromsize)
            probs = (1./float(sromsize))*np.ones((sromsize))
            initial_guess = np.hstack((samples.flatten(), probs))
        else:
            initial_guess = (1./float(sromsize))*np.ones((sromsize))

        return initial_guess
