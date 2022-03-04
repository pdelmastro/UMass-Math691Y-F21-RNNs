"""
Utility functions for numerical integration
"""

import numpy as np

def FE(f, dt, x0, args=None):
    """
    G(x) for Forward Euler Method
    """
    return f(x0, args)


def RK2(f, dt, x0, args=None):
    """
    G(x) for Second order Runge-Kutta
    """
    k1 = f(x0, args)
    k2 = f(x0 + k1 * dt/2, args)
    return k2


def RK4(f, dt, x0, args=None):
    """
    G(x) for Fourth order Runge-Kutta
    """
    k1 = f(x0, args)
    k2 = f(x0 + k1 * dt/2, args)
    k3 = f(x0 + k2 * dt/2, args)
    k4 = f(x0 + dt * k3, args)
    return (k1+k4)/6 + (k2+k3)/3


def select_numerical_method(method):
    """
    Helper function for selecting a numerical method

    Arguments

        method : (String) used to select the solver method.
                        Options: 'FE' for Forward Euler
                                 'RK2' for second-order Runge-Kutta
                                 'RK4' for Fourth-Order Runge-Kutta

    Returns

        G      : Callable function for the selected numerical method
    """
    if method == 'FE':
        return FE
    elif method == 'RK2':
        return RK2
    elif method == 'RK4':
        return RK4
    else:
        raise ValueError('Numerical method %s not recognized.' % method)


def my_odeint(f, dt, T, x0, method='RK2', args=None):
    """
    General numerical integrator where we can specify which solver to us

    Arguments (re-interating what was written above)

        f      : A callable function that computes the vector field
                 for the ODE.

        dt     : Step-size size (float) for the solver. Note this step
                 size is NOT adaptive.

        T      : Maximum time (float). Solver computes x(t_n) for all
                 t_n = n * dt <= T

        x0     : (N,)-numpy array storing the initial condition

        args   :  (Optional tuple) Arguments for the vector field
                ```f```.

        method : (String) used to select the solver method.
                    Options: 'FE' for Forward Euler
                             'RK2' for second-order Runge-Kutta
                             'RK4' for Fourth-Order Runge-Kutta.

    Returns

        x      : (M,N)-numpy array where
                     M = int(T/dt) + 1 is the number of timesteps
                     x[n] = approx for x(n * dt)
    """
    # Select the numerical integration method
    G = select_numerical_method(method)

    # Number of timesteps
    n_ts = int(T/abs(dt)) + 1

    # Array to store the trajectory
    x = np.zeros((n_ts, x0.size))
    x[0] = x0

    # Interatively compute x[t + n * dt]
    for n in range(1, n_ts):
        x[n] = x[n-1] + dt * G(f, dt, x[n-1], args)

    # Return the generated trajectory
    return x
