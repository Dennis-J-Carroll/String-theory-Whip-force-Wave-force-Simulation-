"""
Physical constants and parameters for the whip force simulation.
"""
import numpy as np

# Physical Constants
# Refined based on the balance between gravitational and repulsive forces
# at the critical density of the universe
CRITICAL_DENSITY = 8.5e-27  # kg/m^3
MASS_HYDROGEN = 1.67e-27    # kg
GRAVITATIONAL_CONSTANT = 6.67e-11  # N m^2/kg^2

# Derived Constants
VOLUME_HYDROGEN = MASS_HYDROGEN / CRITICAL_DENSITY
CHARACTERISTIC_LENGTH = VOLUME_HYDROGEN ** (1/3)

# Force Constants
# k1: Repulsive force constant
K1 = 1.35e-66  # Derived from equating repulsive and gravitational forces
# k2: Attractive force constant
K2 = 0.5       # Placeholder value
# Wave speed
C = 1.0

# Simulation Parameters
# Spatial parameters
NX = 500  # Number of spatial points
DX = 0.1  # Spatial step size
X = np.linspace(0, (NX-1)*DX, NX)  # Spatial grid

# Time parameters
DT = 0.01  # Time step size
T = 5.0    # Total simulation time

def create_initial_conditions():
    """
    Creates the initial Gaussian wave packet.
    
    Returns:
        numpy.ndarray: Initial wave displacement
    """
    return np.exp(-(X - 25)**2 / (2 * 5**2))
