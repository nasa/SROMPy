'''
Define SROM-based output surrogate class
'''

import numpy as np

from srom import SROM

class SROMSurrogate:

    def __init__(self, inputsrom, outputsamples, outputgradients=None):
        '''
        Initialize SROM surrogate using the input SROM used to generate the
        output samples. Output gradients are also supplied for the case of
        the piecewise linear surrogate.

        output samples have the following convention:

        m = SROM size (superscript), do = dimension (subscript); 
        
        outputsamples =  |  y^(1)_1, ..., y^(1)_do | ;    
                         |  ...    , ..., ...      |              
                         |  y^(m)_1, ..., y^(m)_do |              

        output samples must match the order of inputsrom samples/probs!!

        (m x d_i array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_di |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_di |

        do - dimension of output samples (doesn't need to equal di of input)

        '''
        
        if (inputsrom._samples is None or inputsrom._probs is None):
            raise ValueError("Input SROM must be properly initialized")

        self._inputsrom = inputsrom


        #Handle 1 dimension case, adjust shape:
        if len(outputsamples.shape) == 1:
            outputsamples.shape = (len(outputsamples), 1)

        #Verify dimensions of samples/probs
        (size, dim) = outputsamples.shape

        if size != self._inputsrom._size:
            raise ValueError("Number of output samples must match input " + 
                                " srom size!")
        
        self._outsamples = outputsamples
        self._dim = dim
        self._size = size

        #TODO - checks on outputgradients:
        if outputgradients is not None:
            (size__, dim__) = outputgradients.shape
            if size__ != self._inputsrom._size:
                raise ValueError("Incorrect # samples in gradient array!")
            if dim__ != self._inputsrom._dim:
                raise ValueError("Incorrect dimension in gradient array!")

        self._gradients = outputgradients

        #Make SROM for output? 
        self._outputsrom = SROM(size, dim)
        self._outputsrom.set_params(outputsamples, inputsrom._probs)

 
    #Do these change for linear surrogate?    
    def compute_moments(self, max_order):
        '''
        Calculate SROM moments up to order 'max_order'. Returns a 
        (max_order x dim) size numpy array with SROM moments for each dimension.
        Computes moments from 1,...,max_order
        '''
        return self._outputsrom.compute_moments(max_order)

    def compute_CDF(self, x_grid):
        '''
        Evaluates the SROM CDF values for each dimension at each point in
        x_grid. x_grid can be a 1D array in which case the CDFs for each random
        vector dimension are evaluated at the same points, or it can be a 
        (num_grid_pts x dim) array, specifying different points for each 
        dimension - each dimension can have a different range of values but
        must have the same # of grid pts across it. Returns a (num_grid_pts x
        dim) array of corresponding CDF values at the grid points

        Note:
             -this is marginal CDFs along each dimension
             -increasing size of x_grid will slow this down a lot

        inputs:
            x_grid, (#grid pts x 1 ) or (#grid pts x dim) array of points to 
                evaluate the SROM CDF at for case of the same grid for each dim
                or different grid, respectively.

        returns:
            cdf_vals, (#grid pts x dim) array of SROM CDF values at x_grid pts
        '''
        return self._outputsrom.compute_CDF(x_grid)

    def sample(self, inputsamples):
        '''
        Generates output samples from the SROM surrogate corresponding to
        the provided input samples.

        inputsamples (numpy array):  (N: # samples, di: input dim, do: out dim)

        inputsamples =  |  x^(1)_1, ..., x^(1)_di | ;    
                        |  ...    , ..., ...      |              
                        |  x^(N)_1, ..., x^(N)_di |  
        
        
        surr_samples =      |  y^(1)_1, ..., y^(1)_do | ;    
                            |  ...    , ..., ...      |              
                            |  y^(N)_1, ..., y^(N)_do |  

        '''
        
        #Handle 1 dimension case, adjust shape:
        if len(inputsamples.shape) == 1:
            inputsamples.shape = (len(inputsamples), 1)

        #Verify dimensions of samples/probs
        (numsamples, dim) = inputsamples.shape
    
        if dim != self._inputsrom._dim:
            raise ValueError("Incorrect input sample dimension")
       
        #Evaluate piecewise constant or linear surrogate model to get samples:
        if self._gradients is None:
            surr_samples = self.sample_pwconstant_surrogate(inputsamples)
        else:
            surr_samples = self.sample_pwlinear_surrogate(inputsamples)

        return surr_samples

    def sample_pwconstant_surrogate(self, inputsamples):
        '''
        Evaluate standard piecewise constant output surrogate model
        '''

        inputsamples_srom = self._inputsrom._samples

        #Generate surrogate samples:
        (numsamples, dim) = inputsamples.shape

        #Generate surrogate samples:
        surr_samples = np.zeros((numsamples, self._dim))
        for i in range(numsamples):
            #Find which input SROM sample is closest to current sample
            sample_i = inputsamples[i,:]
            diff_norms = np.linalg.norm(sample_i - inputsamples_srom, axis=1)
            sromindex = np.argmin(diff_norms)
            surr_samples[i,:] = self._outsamples[sromindex,:]

        return surr_samples

    def sample_pwlinear_surrogate(self, inputsamples): 
        '''
        Evaluate the linear output surrogate model using input SROM samples
        and gradients
        
        input:
        inputsamples =  |  x^(1)_1, ..., x^(1)_di | ;    
                        |  ...    , ..., ...      |              
                        |  x^(N)_1, ..., x^(N)_di | 

        (mxd array)
        gradients = | dy(x^{(1)})/dx_1, ..., dy(x^{(1)})/dx_d |
                    | ...             , ...,    ...           |
                    | dy(x^{(m)})/dx_1, ..., dy(x^{(m)})/dx_d |
    
        '''
        
        inputsamples_srom = self._inputsrom._samples

        #Generate surrogate samples:
        (numsamples, dim) = inputsamples.shape

        #Generate surrogate samples:
        surr_samples = np.zeros((numsamples, self._dim))
        for i in range(numsamples):
            #Find which input SROM sample is closest to current sample
            sample_i = inputsamples[i,:]
            diffs = sample_i - inputsamples_srom
            diff_norms = np.linalg.norm(diffs, axis=1)
            sromindex = np.argmin(diff_norms)

            #Calculate ouput sample value (eq 11b from emery paper)
            output_k = self._outsamples[sromindex,:]
            diffs_k = diffs[sromindex, :]
            grad_k = self._gradients[sromindex, :]
            
            out = output_k + np.dot(grad_k, diffs_k)
            surr_samples[i,:] = out
            

        return surr_samples




        

