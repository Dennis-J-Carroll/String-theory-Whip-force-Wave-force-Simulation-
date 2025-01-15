import numpy as np
import matplotlib.pyplot as plt

# --- Theoretical Framework and Justification ---
# The code simulates a modified wave equation that incorporates a force term derived from a potential function.
# This potential function, V(u), is designed to model interactions between waves in a hypothetical "fabric" of spacetime.
# The concept is inspired by the idea of a "whip force" arising from the collective motion of these waves,
# potentially contributing to phenomena like the accelerating expansion of the universe (dark energy).

# --- Constants ---
# k1 and k2 are crucial parameters that determine the strength of repulsive and attractive forces, respectively.
k1 = 1.0  # Placeholder value for the repulsive force constant.
# TODO: Refine k1 based on the critical density of the universe and the balance between gravitational and repulsive forces.
# Theoretical basis: k1 should be related to the pressure exerted by dark energy, preventing gravitational collapse beyond a certain point.
# Constraint: Set the repulsive force equal to the gravitational force at the critical density of the universe.
# Example: k1 * 12 / u**13 = G * m1 * m2 / u**2, where u is a characteristic length scale related to the average separation of particles at critical density.

k2 = 0.5  # Placeholder value for the attractive force constant.
# TODO: Refine k2 based on the observed acceleration of the universe's expansion.
# Theoretical basis: k2 should relate to the binding forces, potentially contributing to structure formation on smaller scales.
# Constraint: Relate k2 to the acceleration parameter (a) and the equation of state parameter (w) for dark energy.
# Example: k2 * 6 / u**7 is proportional to the second derivative of the scale factor with respect to time.

c = 1.0   # Wave speed. Represents the speed at which disturbances propagate through the hypothetical spacetime fabric.
# This could be related to the speed of light or another fundamental constant.

# --- Potential Function V(u) ---
# This function models the potential energy associated with the interaction of waves.
# It is analogous to the Lennard-Jones potential used in molecular physics.
def V(u):
    """
    Potential function representing the interaction energy between waves.

    Parameters:
        u (float or numpy array): Wave displacement.

    Returns:
        float or numpy array: Potential energy.
    """
    return k1 * (1/u**12) - k2 * (1/u**6)
# Theoretical basis:
# The term k1/u**12 represents a strong repulsive force at small distances, preventing waves from overlapping.
# The term k2/u**6 represents an attractive force at larger distances, potentially contributing to wave binding and structure formation.

# --- Force Function F(u) ---
# This function calculates the force acting on the waves, derived as the negative gradient of the potential function.
def F(u):
    """
    Force function, derived as the negative gradient of the potential function.

    Parameters:
        u (float or numpy array): Wave displacement.

    Returns:
        float or numpy array: Force acting on the wave.
    """
    return 12 * k1 * (1/u**13) - 6 * k2 * (1/u**7)
# Theoretical basis:
# F(u) = -dV/du, representing the force as the rate of change of potential energy with respect to displacement.
# This ensures that the force drives the system towards lower potential energy states.

# --- Wave Equation Solver ---
# This function numerically solves the modified wave equation using a finite difference method.
def wave_equation_solver(u0, dt, dx, T):
    """
    Solves the modified wave equation with a force term using a finite difference method.

    Parameters:
        u0 (numpy array): Initial wave displacement.
        dt (float): Time step.
        dx (float): Spatial step.
        T (float): Total simulation time.

    Returns:
        numpy array: Wave displacement at each time step.
    """
    Nx = len(u0)
    Nt = int(T / dt)
    u = np.zeros((Nt, Nx))
    u[0, :] = u0

    # Initial conditions for the first two time steps
    # We need to initialize two time steps to proceed with the central difference scheme in time.
    for i in range(1, Nx - 1):
        u[1, i] = u[0, i] + 0.5 * dt**2 * (c**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / dx**2 + F(u[0, i]))

    # Time-stepping loop
    # This loop iterates through time, calculating the wave displacement at each step using the finite difference approximation of the wave equation.
    for n in range(1, Nt - 1):
        for i in range(1, Nx - 1):
            # Central difference scheme in space and time.
            # This is a stable and accurate method for solving hyperbolic PDEs like the wave equation.
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + dt**2 * (c**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2 + F(u[n, i]))

    return u

# --- Initial Conditions ---
# These parameters define the initial state of the wave.
Nx = 500  # Number of spatial points.
dx = 0.1  # Spatial step size.
x = np.linspace(0, (Nx-1)*dx, Nx) # Spatial grid.
u0 = np.exp(-(x - 25)**2 / (2 * 5**2))  # Gaussian initial displacement, representing a localized disturbance.
# TODO: Explore different initial conditions to see how they affect the wave dynamics.
# Examples: Multiple Gaussian pulses, sinusoidal waves, random initial displacements.

# --- Time Parameters ---
dt = 0.01 # Time step size.
# The choice of dt is crucial for the stability of the numerical solution.
# It should satisfy the Courant-Friedrichs-Lewy (CFL) condition: c * dt / dx <= 1.
T = 5  # Total simulation time.

# --- Solve the Wave Equation ---
u = wave_equation_solver(u0, dt, dx, T)

# --- Visualization ---
# This section plots the wave displacement at different time steps.
plt.figure(figsize=(10, 6))
for i in range(0, u.shape[0], 50):
    plt.plot(x, u[i, :], label=f't={i*dt:.2f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Wave Propagation with Force Term')
plt.legend()
plt.show()

# --- Further Development ---
# 1. Refine k1 and k2: Investigate how to derive these constants from cosmological observations and physical principles.
# 2. Explore Boundary Conditions: Implement different boundary conditions (e.g., periodic, fixed) to model various physical scenarios.
# 3. 2D/3D Simulation: Extend the simulation to higher dimensions to better represent the complexity of spacetime.
# 4. Chaos and Fuzzy Logic: Incorporate elements of chaos theory and fuzzy logic to account for uncertainties and sensitivities to initial conditions.
# 5. Connection to Observables: Relate the simulation results to observable phenomena like the cosmic microwave background, gravitational waves, and large-scale structure formation.
# 6. Numerical Stability: Further analyze the numerical stability of the finite difference scheme and explore alternative numerical methods if needed.
