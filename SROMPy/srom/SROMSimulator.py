import numpy as np

from SROMPy.srom.Model import Model
from SROMPy.target.RandomVariable import RandomVariable
from SROMPy.srom import SROM, FiniteDifference as FD, SROMSurrogate

class SROMSimulator(object):
    """
    Computes an estimate based on the Stochastic Reduced Order Model
    algorithm.
    """
    def __init__(self, random_input, model):
        #Update docstring with option params (TODO)
        """
        Requires a data object that provides input samples and a model.

        :param random_input: Provides a data sampling function.
        :type random_input: Input
        :param model: A model that outputs from a sample data input.
        :type model: Model
        :param enhanced_optimize: A model that outputs from a sample data input.
        :type enhanced_optimize: bool
        """
        self.__check_init_parameters(random_input, model)

        self._random_variable_data = random_input
        self._model = model
        self._enhanced_optimize = False
        self._weights = None
        self._num_test_samples = 50
        self._error = 'SSE'
        self._max_moment = 5
        self._cdf_grid_pts = 100
        self._tolerance = None
        self._options = None
        self._method = None
        self._joint_opt = False

    def simulate(self, srom_size, dim, surrogate_type, pwl_step_size=None):
        #Read this docstring over (TODO)
        """Performs the SROM Simulation.

        :param srom_size: Size of SROM.
        :type srom_size: int
        :param dim: Dimension of random quantity being modeled.
        :type dim: int
        :param surrogate_type: The SROM type being simulated. Piecewise constant
            and piecewise linear are currently implemented.
        :type surrogate_type: str
        :param pwl_step_size: Step size for piecewise linear, defaults to None.
        :param pwl_step_size: float, optional
        :return: Returns a SROM surrogate object.
        :rtype: SROMSurrogate
        """
        self.__check_simulate_parameters(srom_size,
                                         dim,
                                         surrogate_type,
                                         pwl_step_size)

        input_srom = self._generate_input_srom(srom_size, dim)

        if surrogate_type == "PWC":
            output_samples = \
                self._simulate_piecewise_constant(input_srom)

            output_gradients = None

        elif surrogate_type == "PWL":
            output_samples, output_gradients = \
                self._simulate_piecewise_linear(input_srom, pwl_step_size)

        srom_surrogate = \
            SROMSurrogate(input_srom, output_samples, output_gradients)

        return srom_surrogate
    def set_optimization_parameters(self,
                                    weights=None,
                                    num_tests=50,
                                    error='SSE',
                                    max_moment=5,
                                    cdf_grid_pts=100,
                                    tolerance=None,
                                    options=None,
                                    method=None,
                                    joint_opt=False):
        """
        Sets the additional optimization parameters for SROM's optimize method.
        
        :param weights: relative weights specifying importance of matching
            CDFs, moments, and correlation of the target during optimization.
            Default is equal weights [1,1,1].
        :type weights: 1d Numpy array (length = 3)
        :param num_test_samples: Number of sample sets (iterations) to run
            optimization.
        :type num_test_samples: int
        :param error: Type of error metric to use in objective ("SSE", "MAX",
            "MEAN").
        :type error: string
        :param max_moment: Max number of target moments to consider matching
        :type max_moment: int
        :param cdf_grid_pts: Number of points to evaluate CDF error on
        :type cdf_grid_pts: int
        :param tolerance: Tolerance for scipy optimization algorithm.
        :type tolerance: float
        :param options: Scipy optimization algorithm options, see scipy 
            documentation.
        :type options: dict
        :param method: Method used for scipy optimization, see scipy 
            documentation.
        :type method: string
        :param joint_opt: Flag to optimize jointly for samples & probabilities.
        :type joint_opt: bool
        """
        self._enhanced_optimize = True

        self._weights = weights
        self._num_tests = num_tests
        self._error = error
        self._max_moment = max_moment
        self._cdf_grid_pts = cdf_grid_pts
        self._tolerance = tolerance
        self._method = method
        self._joint_opt = joint_opt
                          
    def _simulate_piecewise_constant(self, input_srom):
        """
        Performs the simulation of the piecewise constant function.

        :param input_srom: SROM that was initialized in simulate method.
        :type input_srom: SROM
        :return: Returns the SROM output samples.
        :rtype: np.ndarray
        """
        srom_output, _ = \
            self.evaluate_model_for_samples(input_srom)

        return srom_output

    def _simulate_piecewise_linear(self, input_srom, pwl_step_size):
        """
        Performs the simulation of the piecewise linear function.

        :param input_srom: SROM that was initialized in simulate method.
        :type input_srom: SROM
        :param pwl_step_size: Step size used to generate the gradient and the
            perturbed samples.
        :type pwl_step_size: float
        """
        srom_output, samples = \
            self.evaluate_model_for_samples(input_srom)

        samples_fd = \
            FD.get_perturbed_samples(samples,
                                     perturbation_values=[pwl_step_size])

        gradient = \
            self._compute_pwl_gradient(srom_output,
                                       samples_fd,
                                       pwl_step_size,
                                       input_srom)

        return srom_output, gradient

    def _compute_pwl_gradient(self, srom_output, samples_fd, step_size,
                              input_srom):
        #Find out more about perturbed samples for docstring (TODO)
        """
        Computes the gradient for the piecewise linear function.

        :param srom_output: Samples generated in _simulate_piecewise_linear
            method.
        :type srom_output: np.ndarray
        :param samples_fd: Perturbed samples.
        :type samples_fd: np.ndarray
        :param step_size: Step sized used to compute the gradient.
        :type step_size: float
        :param input_srom: SROM initialized in simulate function.
        :type input_srom: SROM
        :return: Returns the gradient for SROMSurrogate.
        :rtype: np.ndarray
        """
        perturbed_output, _ = \
            self.evaluate_model_for_samples(input_srom, samples_fd)

        gradient = FD.compute_gradient(srom_output,
                                       perturbed_output,
                                       [step_size])

        return gradient

    def evaluate_model_for_samples(self, input_srom, samples_fd=None):
        """
        Uses model's evaluate method to return output samples.

        :param input_srom: SROM initialized in simulate method.
        :type input_srom: SROM
        :param samples_fd: Perturbed samples used for piecewise linear function,
            defaults to None.
        :param samples_fd: np.ndarray, optional
        :return: Returns output samples generated by model's evaluate method.
        :rtype: np.ndarray
        """
        if samples_fd is not None:
            samples = samples_fd
        else:
            (samples, _) = input_srom.get_params()

        output = np.zeros(len(samples))

        for i, values in enumerate(samples):
            output[i] = self._model.evaluate([values])

        return output, samples

    def _generate_input_srom(self, srom_size, dim):
        """
        Generates an SROM with desired parameters.

        :param srom_size: The size of SROM.
        :type srom_size: int
        :param dim: Dimension of random quantity being modeled.
        :type dim: int
        :return: Returns an SROM object.
        :rtype: SROM
        """
        srom = SROM(srom_size, dim)

        if self._enhanced_optimize == True:
            srom.optimize(target_random_variable=self._random_variable_data,
                          weights=self._weights,
                          num_test_samples=self._num_test_samples,
                          error=self._error,
                          max_moment=self._max_moment,
                          cdf_grid_pts=self._cdf_grid_pts,
                          tolerance=self._tolerance,
                          options=self._options,
                          method=self._method,
                          joint_opt=self._joint_opt)
        else:
            srom.optimize(self._random_variable_data)

        return srom

    @staticmethod
    def __check_init_parameters(data, model):
        """
        Inspects parameters given to the init method.

        :param data: Input object provided to init().
        :param model: Model object provided to init().
        """
        if not isinstance(data, RandomVariable):
            raise TypeError("Data must inherit from the RandomVariable class")

        if not isinstance(model, Model):
            raise TypeError("Model must inherit from Model class")

    @staticmethod
    def __check_simulate_parameters(srom_size, dim, surrogate_type,
                                    pwl_step_size):
        """
        Inspects the parameters given to the simulate method.

        :param srom_size: srom_size input provided to simulate().
        :param dim: dim input provided to simulate().
        :param surrogate_type: surrogate_type provided to simulate().
        :param pwl_step_size: pwl_step_size provided to simulate().
        """
        if not isinstance(srom_size, int):
            raise TypeError("SROM size must be an integer")

        if not isinstance(dim, int):
            raise TypeError("Dim must be an integer")

        if surrogate_type != "PWC" and surrogate_type != "PWL":
            raise ValueError("Surrogate type must be 'PWC' or 'PWL'")
        #Should this be a TypeError or ValueError? Leaning on value(TODO)
        if surrogate_type == "PWL" and pwl_step_size is None:
            raise TypeError("Step size must be initialized for 'PWL' ex: 1e-12")
