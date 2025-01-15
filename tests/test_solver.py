"""
Unit tests for the solver module.
"""
import numpy as np
import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import solver
import constants as const

def test_potential_function():
    """Test the potential function calculation."""
    # Test single value
    assert np.isclose(
        solver.potential_function(1.0),
        const.K1 - const.K2
    )
    
    # Test array
    u = np.array([1.0, 2.0])
    expected = const.K1 * (1/u**12) - const.K2 * (1/u**6)
    np.testing.assert_array_almost_equal(
        solver.potential_function(u),
        expected
    )

def test_force_function():
    """Test the force function calculation."""
    # Test single value
    assert np.isclose(
        solver.force_function(1.0),
        12 * const.K1 - 6 * const.K2
    )
    
    # Test array
    u = np.array([1.0, 2.0])
    expected = 12 * const.K1 * (1/u**13) - 6 * const.K2 * (1/u**7)
    np.testing.assert_array_almost_equal(
        solver.force_function(u),
        expected
    )

def test_cfl_condition():
    """Test the CFL condition checker."""
    assert solver.check_cfl_condition(0.01, 0.1, 1.0) == True
    assert solver.check_cfl_condition(0.2, 0.1, 1.0) == False

def test_wave_equation_solver():
    """Test the wave equation solver."""
    # Test with simple initial conditions
    Nx = 50
    x = np.linspace(0, 5, Nx)
    u0 = 1.0 + 0.1 * np.exp(-(x - 2.5)**2)  # Gaussian pulse with offset

    # Test with valid parameters
    dt, dx, T = 0.01, 0.1, 0.1
    u = solver.wave_equation_solver(u0, dt, dx, T)
    
    assert u.shape == (int(T/dt), len(u0))
    assert np.all(np.isfinite(u))  # Check for no infinities or NaNs
    
    # Test CFL condition violation
    with pytest.raises(ValueError):
        solver.wave_equation_solver(u0, 0.2, 0.1, 0.1)

def test_energy_conservation():
    """Test approximate energy conservation of the system."""
    # Create initial conditions with non-zero offset
    Nx = 100
    x = np.linspace(0, 10, Nx)
    u0 = 1.0 + 0.1 * np.exp(-(x - 5)**2)  # Gaussian pulse with offset
    
    dt, dx, T = 0.01, 0.1, 0.1
    u = solver.wave_equation_solver(u0, dt, dx, T)
    
    # Calculate total energy at start and end
    def calculate_energy(u_slice):
        kinetic = np.sum(np.diff(u_slice)**2) / (2 * dx)
        potential = np.sum(solver.potential_function(u_slice[1:-1]))
        return kinetic + potential
    
    initial_energy = calculate_energy(u[0])
    final_energy = calculate_energy(u[-1])
    
    # Check that energy is conserved within 5%
    relative_error = abs(final_energy - initial_energy) / initial_energy
    assert relative_error < 0.05, f"Energy not conserved. Relative error: {relative_error}"
