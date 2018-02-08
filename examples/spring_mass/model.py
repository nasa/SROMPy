
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


#------------------------------------------------------
#Helper function to use scipy integrator in model class 
def mass_spring(state, t, k, m):
    '''
    Return velocity/acceleration given velocity/position and values for 
    stiffness and mass
    '''

    # unpack the state vector
    x = state[0]
    xd = state[1]

    g = 9.8 # metres per second

    # compute acceleration xdd
    xdd = ((-k*x)/m) + g

    # return the two state derivatives
    return [xd, xdd]

#------------------------------------------------------

class SpringMass_1D(object):
    '''
    Defines Spring Mass model with 1 free param (stiffness of spring, k)
    '''
    def __init__(self, m=1.5, state0=None, time_grid=None):
    
        self._m = m
        
        #Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_grid is None:
            time_grid = np.arange(0.0, 10.0, 0.1)

        self._state0 = state0
        self._t = time_grid

    def simulate(self, k=2.5):
        '''
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid    
        '''

        return odeint(mass_spring, self._state0, self._t, args=(k, self._m))

    def get_max_disp(self, k=2.5):
        '''
        Returns the max displacement over the course of the simulation
        '''

        state = odeint(mass_spring, self._state0, self._t, args=(k, self._m))
        return max(state[:,0])


class SpringMass_2D(object):
    '''
    Defines Spring Mass model with 2 free params (spring stiffness, k & mass, m)
    '''
    def __init__(self, state0=None, time_grid=None):
    
        #Give default initial conditions & time grid if not specified
        if state0 is None:
            state0 = [0.0, 0.0]
        if time_grid is None:
            time_grid = np.arange(0.0, 10.0, 0.1)

        self._state0 = state0
        self._t = time_grid
        
    def simulate(self, k=2.5, m=1.5):
        '''
        Simulate spring mass system for given spring constant. Returns state
        (position, velocity) at all points in time grid    
        '''

        return odeint(mass_spring, self._state0, self._t, args=(k, m))

    def get_max_disp(self, k=2.5, m=1.5):
        '''
        Returns the max displacement over the course of the simulation
        '''

        state = odeint(mass_spring, self._state0, self._t, args=(k, m))
        return max(state[:,0])


if __name__ == '__main__':

    k = 2.5 # Newtons per metre
    m = 1.5 # Kilograms
    state0 = [0.0, 0.0]  #initial conditions
    t = np.arange(0.0, 10.0, 0.1)  #time grid for simulation

    #Initialize model & simulate 
    model = SpringMass_2D(state0, t)
    state = model.simulate(k, m)
    print "shape  = ", state.shape

    #plot results
    plt.figure()
    plt.plot(t, state)
    plt.xlabel('TIME (sec)')
    plt.ylabel('States')
    plt.title('Mass-Spring System')
    plt.legend(('$x$ (m)', '$\dot{x}$ (m/sec)'))
    plt.show()
