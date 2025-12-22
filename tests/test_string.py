"""
Unit tests for the String model.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from string_model import String


def test_string_initialization():
    """Test basic string initialization."""
    string = String(length=50.0, num_points=100, tension=50.0)

    assert string.length == 50.0
    assert string.num_points == 100
    assert string.tension == 50.0
    assert len(string.x) == 100
    assert len(string.displacement) == 100
    assert len(string.velocity) == 100
    assert len(string.density) == 100


def test_uniform_density():
    """Test uniform density profile."""
    string = String(length=50.0, num_points=100, density_profile='uniform', density_uniform=0.01)

    assert np.allclose(string.density, 0.01)

    # Check wave speed is constant
    expected_wave_speed = np.sqrt(50.0 / 0.01)
    assert np.allclose(string.wave_speed, expected_wave_speed)


def test_tapered_density():
    """Test tapered density profile for whip simulation."""
    string = String(
        length=100.0,
        num_points=100,
        density_profile='tapered',
        density_base=0.02,
        density_tip=0.002,
        taper_exponent=1.0
    )

    # Check density decreases from base to tip
    assert string.density[0] > string.density[-1]
    assert np.isclose(string.density[0], 0.02, atol=1e-3)
    assert np.isclose(string.density[-1], 0.002, atol=1e-3)

    # Check wave speed increases from base to tip
    assert string.wave_speed[0] < string.wave_speed[-1]


def test_initial_gaussian():
    """Test Gaussian initial condition."""
    string = String(length=50.0, num_points=500)
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    # Check peak is at center
    center_idx = np.argmax(string.displacement)
    assert np.isclose(string.x[center_idx], 25.0, atol=1.0)

    # Check amplitude
    assert np.isclose(np.max(string.displacement), 1.0, atol=0.01)

    # Check velocity is zero
    assert np.allclose(string.velocity, 0.0)


def test_initial_sine():
    """Test sine wave initial condition."""
    string = String(length=50.0, num_points=500)
    string.set_initial_sine(wavelength=10.0, amplitude=0.5)

    # Check amplitude
    assert np.isclose(np.max(string.displacement), 0.5, atol=0.01)
    assert np.isclose(np.min(string.displacement), -0.5, atol=0.01)


def test_energy_calculation():
    """Test kinetic and potential energy calculations."""
    string = String(length=50.0, num_points=500, tension=50.0, density_profile='uniform')

    # Zero initial state should have zero energy
    assert np.isclose(string.get_kinetic_energy(), 0.0, atol=1e-10)
    assert np.isclose(string.get_potential_energy(), 0.0, atol=1e-10)

    # Set non-zero displacement
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    # Should have potential energy but no kinetic energy
    pe = string.get_potential_energy()
    ke = string.get_kinetic_energy()
    assert pe > 0
    assert np.isclose(ke, 0.0, atol=1e-10)

    # Add velocity
    string.velocity = np.ones_like(string.velocity) * 0.1
    ke_with_velocity = string.get_kinetic_energy()
    assert ke_with_velocity > 0


def test_boundary_conditions_fixed():
    """Test fixed boundary conditions."""
    string = String(length=50.0, num_points=100, boundary_left='fixed', boundary_right='fixed')

    # Set some displacement
    string.displacement = np.ones(100)
    string.velocity = np.ones(100) * 0.5

    # Apply boundary conditions
    string.apply_boundary_conditions()

    # Check boundaries are fixed
    assert string.displacement[0] == 0.0
    assert string.displacement[-1] == 0.0
    assert string.velocity[0] == 0.0
    assert string.velocity[-1] == 0.0


def test_boundary_conditions_free():
    """Test free boundary conditions."""
    string = String(length=50.0, num_points=100, boundary_left='free', boundary_right='free')

    # Set some displacement
    string.displacement = np.linspace(0, 1, 100)

    # Apply boundary conditions
    string.apply_boundary_conditions()

    # For free boundaries, derivative should be zero (approximately)
    # This means boundary value should equal its neighbor
    assert np.isclose(string.displacement[0], string.displacement[1])
    assert np.isclose(string.displacement[-1], string.displacement[-2])


def test_get_max_wave_speed():
    """Test maximum wave speed calculation."""
    string = String(length=50.0, num_points=100, tension=100.0, density_profile='tapered',
                   density_base=0.02, density_tip=0.002)

    max_speed = string.get_max_wave_speed()

    # Max speed should be at the tip (minimum density)
    expected_max_speed = np.sqrt(100.0 / 0.002)
    assert np.isclose(max_speed, expected_max_speed, rtol=0.01)


def test_state_get_set():
    """Test getting and setting string state."""
    string = String(length=50.0, num_points=100, boundary_left='free', boundary_right='free')
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    # Get state
    disp, vel = string.get_state()

    # Modify state
    disp *= 2
    vel += 0.5

    # Set new state
    string.set_state(disp, vel)

    # Check state was updated (approximately, since boundaries may be adjusted)
    # Check interior points more strictly
    assert np.allclose(string.displacement[1:-1], disp[1:-1])
    assert np.allclose(string.velocity[1:-1], vel[1:-1])


def test_custom_density_function():
    """Test custom density profile."""
    def custom_density(x):
        # Parabolic density profile
        return 0.01 * (1 + (x / 50.0)**2)

    string = String(length=50.0, num_points=100, density_profile='custom',
                   custom_density_func=custom_density)

    # Check density matches custom function
    expected_density = custom_density(string.x)
    assert np.allclose(string.density, expected_density)
