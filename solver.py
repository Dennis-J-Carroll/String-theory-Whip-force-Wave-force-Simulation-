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

        # Free right boundary: use a mirrored ghost node (u[N+1] = u[N-1]) so
        # the tip node experiences real restoring forces. Without this the tip
        # has zero acceleration forever and a whip could never crack.
        if self.string is not None and getattr(self.string, "boundary_right", "fixed") == "free":
            d2u_dx2[-1] = 2.0 * (u[-2] - u[-1]) / dx**2

        acceleration = c**2 * d2u_dx2

        # Add external force if enabled
        if self.enable_force:
            force = force_function(u, self.k1, self.k2)
            # Numerical guard: F(u) ~ 1/u^13 diverges as u -> 0, which used to
            # detonate the solution at step 1 (energies ~1e67). A generous cap
            # never engages in normal dynamics — only at the singularity.
            force = np.clip(force, -1e4, 1e4)
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
            # First step: second-order Taylor start, u(t-dt) = u - v*dt + 0.5*a*dt^2.
            # The naive u - v*dt is only first-order accurate and would cap the
            # entire scheme's convergence at O(dt) (the initial condition itself
            # would inject a larger error than the integration ever removes).
            self.u_prev = u - v * dt + 0.5 * dt**2 * self.compute_acceleration(u, c, dx)

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


# ============================================================================
# Compatibility shim — the wave_solver(x, u0, u0_prev, c, dt, T, K1, K2) API
# ============================================================================

# Wave-scale Lennard-Jones calibration (identical to main.py's): the well is
# expressed at meter scale instead of the cosmological 1e-66 constants, with
# equilibrium u* = 2 m and soft-well curvature omega0 = 5 rad/s. k1 = k2*u*^6/2
# places the equilibrium exactly at u* (F(u*) = 0).
WELL_U_STAR = 2.0
WELL_OMEGA0 = 5.0
WAVE_SCALE_K2 = WELL_OMEGA0**2 * WELL_U_STAR**8 / 36.0   # ~177.8
WAVE_SCALE_K1 = WAVE_SCALE_K2 * WELL_U_STAR**6 / 2.0     # ~5689.6


def wave_solver(x, u0, u0_prev, c, dt, total_time, K1=None, K2=None,
                fixed_ends=False, save_every=1, return_velocity=False):
    """Run a wave simulation with the legacy positional API.

    ``dashboard_app.py`` and ``interactive_demo.py`` call
    ``wave_solver(x, u0, u0_prev, c, dt, total_time, K1, K2)``. This shim
    adapts that call to the modern physics: a :class:`String` on the given
    grid integrated with the symplectic Verlet solver (the same scheme the
    main simulation uses), so every entry point in the repo shares one
    force model — including the force clamp at the 1/u^13 singularity.

    Args:
        x: Spatial grid (np.ndarray). Must be evenly spaced.
        u0: Initial displacement (np.ndarray, same shape as x).
        u0_prev: Displacement one step earlier (or None for a rest start).
        c: Wave speed (scalar).
        dt: Time step (s).
        total_time: Duration to integrate (s).
        K1: Repulsive force constant (None -> calibrated wave-scale default).
        K2: Attractive force constant (None -> calibrated wave-scale default).
        fixed_ends: Pin both endpoints at u = 0. Defaults to False (free,
            Neumann ends) because the string rests at the well equilibrium
            u* = 2 m: clamping the ends to zero fights the well forever,
            doing net work on the string and wrecking energy conservation.
        save_every: Keep a frame every N solver steps.
        return_velocity: If True, return the solver's exact stored
            velocities at each saved frame as a third array — use these for
            kinetic energy instead of reconstructing from frames (the
            reconstruction is exact at interior frames but finite-order at
            the endpoints).

    Returns:
        (u_history, time_points): displacement frames (np.ndarray, one per
        saved step, including t = 0) and matching times (np.ndarray).
        With ``return_velocity=True``: (u_history, time_points, v_history).
    """
    if String is None:  # pragma: no cover - string_model is a core module
        raise ImportError("string_model.String unavailable")
    if dt <= 0:
        raise ValueError("dt must be positive")

    x = np.asarray(x, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-6, atol=1e-12):
        raise ValueError("wave_solver requires an evenly spaced grid")

    k1 = float(K1) if K1 is not None else WAVE_SCALE_K1
    k2 = float(K2) if K2 is not None else WAVE_SCALE_K2

    length = float(x[-1] - x[0])
    string = String(
        length=length,
        num_points=len(x),
        tension=float(c) ** 2,            # tension = rho * c^2 (rho = 1)
        density_profile="uniform",
        density_uniform=1.0,              # rho = 1 so tension = c^2 exactly
        boundary_left="fixed" if fixed_ends else "free",
        boundary_right="fixed" if fixed_ends else "free",
        # NOTE: the default is free/free — see fixed_ends in the docstring:
        # the string rests at u* = 2 m, and clamping the ends to zero fights
        # the well (the boundary does net work, wrecking conservation).
    )

    # Rest-position background: place the string at its well equilibrium.
    # Without this, the calibrated force pulls the whole string toward u*,
    # turning the caller's IC into a decaying oscillation about nothing.
    u_star = (2.0 * k1 / k2) ** (1.0 / 6.0)
    string.displacement = u_star + u0.copy()
    if u0_prev is not None:
        u_prev = np.asarray(u0_prev, dtype=float)
        string.velocity = (u0 - u_prev) / dt
    else:
        string.velocity = np.zeros_like(u0)

    solver = VerletSolver(string, enable_force=True, k1=k1, k2=k2,
                          damping=0.0)
    solver.solve(total_time, dt, save_interval=max(1, int(save_every)),
                 check_cfl=True, verbose=False)

    u_history = np.array(solver.displacement_history)
    time_points = np.array(solver.time_history)
    if return_velocity:
        # velocity_history is saved in lockstep with displacement_history
        # (including the exact v0 at t = 0), so it is already frame-aligned.
        v_history = np.array(solver.velocity_history)
        return u_history, time_points, v_history
    return u_history, time_points


def saved_frame_velocities(u_history, time_points, u0_prev=None, v0=None):
    """Velocities of saved frames, honest to the leapfrog recurrence.

    ``np.gradient`` over the saved frames already reproduces the exact
    central-difference relation v[n] = (u[n+1] - u[n-1]) / (2*dt) at every
    interior frame (uniform dt) — but it falls back to one-sided differences
    at the endpoints. The first frame's velocity is known exactly from the
    initial condition, so it is substituted; the last frame keeps its
    one-sided estimate (error O(a*dt), one frame out of thousands).

    Args:
        u_history: Saved displacement frames.
        time_points: Matching times (uniform dt).
        u0_prev: Displacement one step before frame 0, in the SAME
            coordinates as u_history (i.e. including any background the
            shim added). Alternative to ``v0``.
        v0: Exact initial velocity, preferred over ``u0_prev``.

    Returns an array of the same shape as ``u_history``.
    """
    u_history = np.asarray(u_history, dtype=float)
    time_points = np.asarray(time_points, dtype=float)
    # edge_order=2: second-order one-sided differences at the end frames.
    # The default (1st order) leaves an O(a*dt) velocity error on the final
    # frame, which then dominates the whole energy-drift budget.
    v = np.gradient(u_history, time_points, axis=0, edge_order=2)
    if len(u_history) > 1:
        if v0 is not None:
            v[0] = v0
        elif u0_prev is not None:
            v[0] = (u_history[0] - np.asarray(u0_prev, dtype=float)) / (
                time_points[1] - time_points[0])
        else:
            v[0] = 0.0
    return v
