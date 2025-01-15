"""
Numerical solvers and core functions for the whip force simulation.
"""
import numpy as np
import constants as const

def potential_function(u):
    """
    Potential function representing the interaction energy between waves.

    Parameters:
        u (float or numpy.ndarray): Wave displacement

    Returns:
        float or numpy.ndarray: Potential energy
    """
    # Add small epsilon to prevent division by zero
    u = np.maximum(u, 1e-10)
    return const.K1 * (1/u**12) - const.K2 * (1/u**6)

def force_function(u):
    """
    Force function, derived as the negative gradient of the potential function.

    Parameters:
        u (float or numpy.ndarray): Wave displacement

    Returns:
        float or numpy.ndarray: Force acting on the wave
    """
    # Add small epsilon to prevent division by zero
    u = np.maximum(u, 1e-10)
    return 12 * const.K1 * (1/u**13) - 6 * const.K2 * (1/u**7)

def check_cfl_condition(dt, dx, c):
    """
    Check if the CFL condition is satisfied.

    Parameters:
        dt (float): Time step
        dx (float): Spatial step
        c (float): Wave speed

    Returns:
        bool: True if CFL condition is satisfied
    """
    return c * dt / dx <= 1

def wave_equation_solver(u0, dt, dx, T):
    """
    Solves the modified wave equation with a force term using finite difference method.

    Parameters:
        u0 (numpy.ndarray): Initial wave displacement
        dt (float): Time step
        dx (float): Spatial step
        T (float): Total simulation time

    Returns:
        numpy.ndarray: Wave displacement at each time step
    
    Raises:
        ValueError: If CFL condition is not satisfied
    """
    if not check_cfl_condition(dt, dx, const.C):
        raise ValueError("CFL condition not satisfied. Reduce dt or increase dx.")

    Nx = len(u0)
    Nt = int(T / dt)
    u = np.zeros((Nt, Nx))
    u[0, :] = u0

    # Initialize first two time steps
    for i in range(1, Nx - 1):
        u[1, i] = u[0, i] + 0.5 * dt**2 * (
            const.C**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / dx**2 + 
            force_function(u[0, i])
        )

    # Time-stepping loop
    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            u[n+1, i] = (
                2*u[n, i] - u[n-1, i] + 
                dt**2 * (const.C**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2 + 
                force_function(u[n, i]))
            )

    return u
