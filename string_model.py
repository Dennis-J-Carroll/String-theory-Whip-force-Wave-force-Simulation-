"""
String model class for wave simulations.

This module provides an object-oriented representation of a vibrating string
with various density profiles, boundary conditions, and initial conditions.
"""
import numpy as np
from typing import Literal, Optional, Callable


class String:
    """
    Represents a 1D vibrating string with customizable properties.

    Attributes:
        length (float): Total length of the string
        num_points (int): Number of spatial discretization points
        tension (float): Tension in the string (N)
        x (np.ndarray): Spatial grid points
        dx (float): Spatial step size
        displacement (np.ndarray): Current displacement u(x, t)
        velocity (np.ndarray): Current velocity ∂u/∂t
        density (np.ndarray): Linear mass density μ(x) at each point
        wave_speed (np.ndarray): Wave speed c(x) = sqrt(T/μ(x))
    """

    def __init__(
        self,
        length: float = 50.0,
        num_points: int = 500,
        tension: float = 50.0,
        density_profile: Literal["uniform", "tapered", "custom"] = "uniform",
        density_uniform: float = 0.01,
        density_base: float = 0.02,
        density_tip: float = 0.002,
        taper_exponent: float = 1.0,
        boundary_left: Literal["fixed", "free", "periodic"] = "fixed",
        boundary_right: Literal["fixed", "free", "periodic"] = "fixed",
        custom_density_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize a String object.

        Args:
            length: Total length of the string
            num_points: Number of spatial discretization points
            tension: Tension in the string (N)
            density_profile: Type of density distribution
            density_uniform: Uniform density value (kg/m)
            density_base: Density at base for tapered profile (kg/m)
            density_tip: Density at tip for tapered profile (kg/m)
            taper_exponent: Exponent controlling taper rate
            boundary_left: Left boundary condition
            boundary_right: Right boundary condition
            custom_density_func: Custom function for density profile
        """
        self.length = length
        self.num_points = num_points
        self.tension = tension
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right

        # Create spatial grid
        self.x = np.linspace(0, length, num_points)
        self.dx = length / (num_points - 1)

        # Initialize displacement and velocity
        self.displacement = np.zeros(num_points)
        self.velocity = np.zeros(num_points)

        # Set up density profile
        self.density_profile = density_profile
        if density_profile == "uniform":
            self.density = np.ones(num_points) * density_uniform
        elif density_profile == "tapered":
            # Tapered density: decreases from base to tip
            # μ(x) = μ_base * (1 - (1 - μ_tip/μ_base) * (x/L)^n)
            normalized_x = self.x / length
            ratio = density_tip / density_base
            self.density = density_base * (1 - (1 - ratio) * normalized_x**taper_exponent)
            # Ensure minimum density to avoid singularities
            self.density = np.maximum(self.density, 1e-10)
        elif density_profile == "custom" and custom_density_func is not None:
            self.density = custom_density_func(self.x)
            self.density = np.maximum(self.density, 1e-10)
        else:
            raise ValueError(f"Invalid density profile: {density_profile}")

        # Calculate wave speed c(x) = sqrt(T/μ(x))
        self.wave_speed = np.sqrt(tension / self.density)

        # Store additional parameters
        self._density_params = {
            "uniform": density_uniform,
            "base": density_base,
            "tip": density_tip,
            "exponent": taper_exponent,
        }

    def set_initial_gaussian(
        self, center: float = None, width: float = 5.0, amplitude: float = 1.0
    ) -> None:
        """
        Set initial displacement to a Gaussian pulse.

        Args:
            center: Center position (defaults to middle of string)
            width: Standard deviation of Gaussian
            amplitude: Peak amplitude
        """
        if center is None:
            center = self.length / 2

        self.displacement = amplitude * np.exp(-((self.x - center) ** 2) / (2 * width**2))
        self.velocity = np.zeros_like(self.displacement)

    def set_initial_sine(
        self, wavelength: float = 10.0, amplitude: float = 0.5, phase: float = 0.0
    ) -> None:
        """
        Set initial displacement to a sine wave.

        Args:
            wavelength: Wavelength of the sine wave
            amplitude: Amplitude of the sine wave
            phase: Phase offset in radians
        """
        k = 2 * np.pi / wavelength
        self.displacement = amplitude * np.sin(k * self.x + phase)
        self.velocity = np.zeros_like(self.displacement)

    def set_initial_pulse(self, position: float, amplitude: float = 2.0, width: float = 2.0) -> None:
        """
        Set initial displacement to a localized pulse.

        Args:
            position: Position of the pulse
            amplitude: Amplitude of the pulse
            width: Width of the pulse
        """
        # Use a squared exponential for a more localized pulse
        self.displacement = amplitude * np.exp(-((self.x - position) ** 2) / width**2)
        self.velocity = np.zeros_like(self.displacement)

    def set_initial_custom(
        self,
        displacement_func: Callable[[np.ndarray], np.ndarray],
        velocity_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """
        Set custom initial conditions.

        Args:
            displacement_func: Function that takes x array and returns displacement
            velocity_func: Function that takes x array and returns velocity (optional)
        """
        self.displacement = displacement_func(self.x)
        if velocity_func is not None:
            self.velocity = velocity_func(self.x)
        else:
            self.velocity = np.zeros_like(self.displacement)

    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to displacement and velocity."""
        # Left boundary
        if self.boundary_left == "fixed":
            self.displacement[0] = 0.0
            self.velocity[0] = 0.0
        elif self.boundary_left == "free":
            # Free boundary: ∂u/∂x = 0 at boundary
            # Approximate with second-order finite difference
            self.displacement[0] = self.displacement[1]

        # Right boundary
        if self.boundary_right == "fixed":
            self.displacement[-1] = 0.0
            self.velocity[-1] = 0.0
        elif self.boundary_right == "free":
            self.displacement[-1] = self.displacement[-2]

        # Periodic boundaries
        if self.boundary_left == "periodic" and self.boundary_right == "periodic":
            # Ensure continuity
            avg = (self.displacement[0] + self.displacement[-1]) / 2
            self.displacement[0] = avg
            self.displacement[-1] = avg
            avg_vel = (self.velocity[0] + self.velocity[-1]) / 2
            self.velocity[0] = avg_vel
            self.velocity[-1] = avg_vel

    def get_kinetic_energy(self) -> float:
        """
        Calculate total kinetic energy of the string.

        Returns:
            Total kinetic energy: KE = (1/2) * ∫ μ(x) * (∂u/∂t)^2 dx
        """
        # Trapezoidal integration
        integrand = 0.5 * self.density * self.velocity**2
        return np.trapezoid(integrand, dx=self.dx)

    def get_potential_energy(self) -> float:
        """
        Calculate potential energy (elastic strain energy) of the string.

        Returns:
            Total potential energy: PE = (1/2) * T * ∫ (∂u/∂x)^2 dx
        """
        # Calculate spatial derivative using central differences
        du_dx = np.gradient(self.displacement, self.dx)
        integrand = 0.5 * self.tension * du_dx**2
        return np.trapezoid(integrand, dx=self.dx)

    def get_total_energy(self) -> float:
        """
        Calculate total mechanical energy of the string.

        Returns:
            Total energy = Kinetic + Potential
        """
        return self.get_kinetic_energy() + self.get_potential_energy()

    def get_max_wave_speed(self) -> float:
        """
        Get maximum wave speed on the string.

        Returns:
            Maximum value of c(x) = sqrt(T/μ(x))
        """
        return np.max(self.wave_speed)

    def get_tip_velocity(self) -> float:
        """
        Get the velocity at the tip (right end) of the string.

        Useful for whip simulations to track supersonic motion.

        Returns:
            Velocity at the tip
        """
        return self.velocity[-1]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"String(length={self.length}, "
            f"num_points={self.num_points}, "
            f"tension={self.tension}, "
            f"density_profile='{self.density_profile}', "
            f"boundaries=[{self.boundary_left}, {self.boundary_right}])"
        )

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get current state of the string.

        Returns:
            Tuple of (displacement, velocity) arrays
        """
        return self.displacement.copy(), self.velocity.copy()

    def set_state(self, displacement: np.ndarray, velocity: np.ndarray) -> None:
        """
        Set the state of the string.

        Args:
            displacement: Displacement array
            velocity: Velocity array
        """
        self.displacement = displacement.copy()
        self.velocity = velocity.copy()
        self.apply_boundary_conditions()
