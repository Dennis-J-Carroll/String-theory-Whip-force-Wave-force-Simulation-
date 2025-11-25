"""
Numerical solvers and core functions for the wave simulation.

This module provides multiple numerical integration schemes for solving
the wave equation with external forces, including:
- Central Difference (explicit)
- Runge-Kutta 4th order (RK4)
- Velocity Verlet (symplectic)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import constants as const
try:
    from string_model import String
except ImportError:
    String = None  # For backward compatibility


def potential_function(u, k1=None, k2=None):
    """
    Potential function representing the interaction energy between waves.

    Parameters:
        u (float or numpy.ndarray): Wave displacement
        k1 (float): Repulsive force constant (defaults to const.K1)
        k2 (float): Attractive force constant (defaults to const.K2)

    Returns:
        float or numpy.ndarray: Potential energy
    """
    if k1 is None:
        k1 = const.K1
    if k2 is None:
        k2 = const.K2

    # Add small epsilon to prevent division by zero
    u = np.maximum(np.abs(u), 1e-10)
    return k1 * (1/u**12) - k2 * (1/u**6)


def force_function(u, k1=None, k2=None):
    """
    Force function, derived as the negative gradient of the potential function.

    Parameters:
        u (float or numpy.ndarray): Wave displacement
        k1 (float): Repulsive force constant (defaults to const.K1)
        k2 (float): Attractive force constant (defaults to const.K2)

    Returns:
        float or numpy.ndarray: Force acting on the wave
    """
    if k1 is None:
        k1 = const.K1
    if k2 is None:
        k2 = const.K2

    # Add small epsilon to prevent division by zero
    u_safe = np.maximum(np.abs(u), 1e-10)
    return 12 * k1 * (1/u_safe**13) - 6 * k2 * (1/u_safe**7)


def check_cfl_condition(dt, dx, c):
    """
    Check if the CFL condition is satisfied.

    Parameters:
        dt (float): Time step
        dx (float): Spatial step
        c (float): Wave speed (or max wave speed if array)

    Returns:
        bool: True if CFL condition is satisfied
    """
    if isinstance(c, np.ndarray):
        c = np.max(c)
    return c * dt / dx <= 1


def wave_equation_solver(u0, dt, dx, T, c=None):
    """
    Legacy function: Solves the modified wave equation using central difference.

    DEPRECATED: Use CentralDifferenceSolver class instead.

    Parameters:
        u0 (numpy.ndarray): Initial wave displacement
        dt (float): Time step
        dx (float): Spatial step
        T (float): Total simulation time
        c (float): Wave speed (defaults to const.C)

    Returns:
        numpy.ndarray: Wave displacement at each time step
    """
    if c is None:
        c = const.C

    if not check_cfl_condition(dt, dx, c):
        raise ValueError("CFL condition not satisfied. Reduce dt or increase dx.")

    Nx = len(u0)
    Nt = int(T / dt)
    u = np.zeros((Nt, Nx))
    u[0, :] = u0

    # Initialize first time step (vectorized)
    u[1, 1:-1] = u[0, 1:-1] + 0.5 * dt**2 * (
        c**2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / dx**2 +
        force_function(u[0, 1:-1])
    )

    # Time-stepping loop (vectorized spatial operations)
    for n in range(1, Nt - 1):
        u[n+1, 1:-1] = (
            2*u[n, 1:-1] - u[n-1, 1:-1] +
            dt**2 * (c**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]) / dx**2 +
            force_function(u[n, 1:-1]))
        )

    return u


# ============================================================================
# Object-Oriented Solver Classes
# ============================================================================

class WaveSolver(ABC):
    """
    Abstract base class for wave equation solvers.

    All solvers should inherit from this class and implement the step() method.
    """

    def __init__(
        self,
        string: Optional['String'] = None,
        enable_force: bool = True,
        k1: float = None,
        k2: float = None,
        damping: float = 0.0,
    ):
        """
        Initialize the solver.

        Args:
            string: String object to solve (if None, must provide state manually)
            enable_force: Whether to include external force term
            k1: Repulsive force constant
            k2: Attractive force constant
            damping: Artificial damping coefficient (0 = no damping)
        """
        self.string = string
        self.enable_force = enable_force
        self.k1 = k1 if k1 is not None else const.K1
        self.k2 = k2 if k2 is not None else const.K2
        self.damping = damping

        # History tracking
        self.time_history: List[float] = []
        self.displacement_history: List[np.ndarray] = []
        self.velocity_history: List[np.ndarray] = []
        self.energy_history: List[Tuple[float, float, float]] = []

    @abstractmethod
    def step(self, u: np.ndarray, v: np.ndarray, dt: float, dx: float,
             c: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one time step of the integration.

        Args:
            u: Displacement array
            v: Velocity array
            dt: Time step
            dx: Spatial step
            c: Wave speed (scalar or array)

        Returns:
            Tuple of (new_displacement, new_velocity)
        """
        pass

    def compute_spatial_derivative_second(self, u: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute second spatial derivative using central differences.

        ∂²u/∂x² ≈ (u[i+1] - 2*u[i] + u[i-1]) / dx²

        Args:
            u: Displacement array
            dx: Spatial step

        Returns:
            Second derivative array (same shape as u)
        """
        d2u_dx2 = np.zeros_like(u)
        d2u_dx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        return d2u_dx2

    def compute_acceleration(self, u: np.ndarray, c: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute acceleration: ∂²u/∂t² = c² ∂²u/∂x² + F(u)

        Args:
            u: Displacement array
            c: Wave speed (scalar or array)
            dx: Spatial step

        Returns:
            Acceleration array
        """
        # Wave equation term
        d2u_dx2 = self.compute_spatial_derivative_second(u, dx)
        acceleration = c**2 * d2u_dx2

        # Add external force if enabled
        if self.enable_force:
            force = force_function(u, self.k1, self.k2)
            acceleration += force

        return acceleration

    def solve(
        self,
        total_time: float,
        dt: float,
        save_interval: int = 1,
        check_cfl: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Solve the wave equation over the specified time period.

        Args:
            total_time: Total simulation time
            dt: Time step
            save_interval: Save state every N steps (1 = save all)
            check_cfl: Whether to check CFL condition
            verbose: Print progress messages

        Returns:
            Array of shape (num_saved_steps, num_points) with displacement history
        """
        if self.string is None:
            raise ValueError("No string object provided to solver")

        # Check CFL condition
        if check_cfl:
            max_c = self.string.get_max_wave_speed()
            if not check_cfl_condition(dt, self.string.dx, max_c):
                cfl_ratio = max_c * dt / self.string.dx
                raise ValueError(
                    f"CFL condition violated: c*dt/dx = {cfl_ratio:.3f} > 1. "
                    f"Reduce dt to < {self.string.dx / max_c:.6f}"
                )

        # Initialize
        num_steps = int(total_time / dt)
        u = self.string.displacement.copy()
        v = self.string.velocity.copy()

        # Storage
        self.time_history = []
        self.displacement_history = []
        self.velocity_history = []
        self.energy_history = []

        # Save initial state
        self.time_history.append(0.0)
        self.displacement_history.append(u.copy())
        self.velocity_history.append(v.copy())

        # Time stepping
        for n in range(num_steps):
            # Perform one step
            u, v = self.step(u, v, dt, self.string.dx, self.string.wave_speed)

            # Apply boundary conditions
            self.string.displacement = u
            self.string.velocity = v
            self.string.apply_boundary_conditions()
            u = self.string.displacement.copy()
            v = self.string.velocity.copy()

            # Save state if needed
            if (n + 1) % save_interval == 0:
                current_time = (n + 1) * dt
                self.time_history.append(current_time)
                self.displacement_history.append(u.copy())
                self.velocity_history.append(v.copy())

                # Track energy
                ke = self.string.get_kinetic_energy()
                pe = self.string.get_potential_energy()
                self.energy_history.append((ke, pe, ke + pe))

            # Progress reporting
            if verbose and (n + 1) % (num_steps // 10) == 0:
                progress = 100 * (n + 1) / num_steps
                print(f"Progress: {progress:.0f}%")

        # Update string state
        self.string.displacement = u
        self.string.velocity = v

        if verbose:
            print("Simulation complete!")

        return np.array(self.displacement_history)


class CentralDifferenceSolver(WaveSolver):
    """
    Central difference solver for the wave equation.

    Uses explicit finite difference scheme:
    u[n+1] = 2*u[n] - u[n-1] + dt² * acceleration
    """

    def __init__(self, string=None, **kwargs):
        super().__init__(string, **kwargs)
        self.u_prev = None  # Store previous time step

    def step(self, u, v, dt, dx, c, **kwargs):
        """Perform one central difference time step."""
        if self.u_prev is None:
            # First step: use velocity to estimate u_prev
            self.u_prev = u - v * dt

        # Compute acceleration
        acceleration = self.compute_acceleration(u, c, dx)

        # Add damping: -2γv
        if self.damping > 0:
            acceleration -= 2 * self.damping * v

        # Central difference update
        u_new = 2*u - self.u_prev + dt**2 * acceleration

        # Update velocity (central difference in time)
        v_new = (u_new - self.u_prev) / (2 * dt)

        # Update history
        self.u_prev = u.copy()

        return u_new, v_new


class RK4Solver(WaveSolver):
    """
    4th-order Runge-Kutta solver for the wave equation.

    More accurate than central difference but ~4x slower.
    Converts 2nd order ODE to system of 1st order ODEs.
    """

    def step(self, u, v, dt, dx, c, **kwargs):
        """Perform one RK4 time step."""
        # RK4 for system: du/dt = v, dv/dt = acceleration

        # k1
        k1_u = v
        k1_v = self.compute_acceleration(u, c, dx)
        if self.damping > 0:
            k1_v -= 2 * self.damping * v

        # k2
        u_temp = u + 0.5 * dt * k1_u
        v_temp = v + 0.5 * dt * k1_v
        k2_u = v_temp
        k2_v = self.compute_acceleration(u_temp, c, dx)
        if self.damping > 0:
            k2_v -= 2 * self.damping * v_temp

        # k3
        u_temp = u + 0.5 * dt * k2_u
        v_temp = v + 0.5 * dt * k2_v
        k3_u = v_temp
        k3_v = self.compute_acceleration(u_temp, c, dx)
        if self.damping > 0:
            k3_v -= 2 * self.damping * v_temp

        # k4
        u_temp = u + dt * k3_u
        v_temp = v + dt * k3_v
        k4_u = v_temp
        k4_v = self.compute_acceleration(u_temp, c, dx)
        if self.damping > 0:
            k4_v -= 2 * self.damping * v_temp

        # Combine
        u_new = u + (dt / 6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_new = v + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return u_new, v_new


class VerletSolver(WaveSolver):
    """
    Velocity Verlet integrator for the wave equation.

    Symplectic integrator that conserves energy well.
    Good balance between accuracy and speed.
    """

    def step(self, u, v, dt, dx, c, **kwargs):
        """Perform one Verlet time step."""
        # Current acceleration
        a = self.compute_acceleration(u, c, dx)
        if self.damping > 0:
            a -= 2 * self.damping * v

        # Update position
        u_new = u + v * dt + 0.5 * a * dt**2

        # Compute new acceleration
        a_new = self.compute_acceleration(u_new, c, dx)
        if self.damping > 0:
            # For damped Verlet, use average velocity
            v_pred = v + a * dt
            a_new -= 2 * self.damping * v_pred

        # Update velocity
        v_new = v + 0.5 * (a + a_new) * dt

        return u_new, v_new
