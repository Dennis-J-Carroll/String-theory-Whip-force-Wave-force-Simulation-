"""
Unit tests for energy conservation and analysis functions.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from string_model import String
from solver import CentralDifferenceSolver, RK4Solver, VerletSolver
import analysis


def test_energy_conservation_verlet():
    """Test energy conservation with Verlet integrator."""
    string = String(length=50.0, num_points=500, tension=50.0, density_profile='uniform')
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=0.5)

    solver = VerletSolver(string, enable_force=False, damping=0.0)
    solver.solve(total_time=2.0, dt=0.001, save_interval=50, verbose=False)

    # Analyze energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history,
        solver.energy_history,
        tolerance=0.05,
        plot=False
    )

    # Verlet should conserve energy well
    assert energy_stats['is_conserved'], f"Energy not conserved: drift = {energy_stats['relative_drift']*100:.2f}%"


def test_energy_conservation_rk4():
    """Test energy conservation with RK4 integrator."""
    string = String(length=50.0, num_points=500, tension=50.0, density_profile='uniform')
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=0.5)

    solver = RK4Solver(string, enable_force=False, damping=0.0)
    # RK4 needs small dt too
    solver.solve(total_time=1.0, dt=0.0005, save_interval=100, verbose=False)

    # Analyze energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history,
        solver.energy_history,
        tolerance=0.05,
        plot=False
    )

    # RK4 should also conserve energy well
    assert energy_stats['is_conserved'], f"Energy not conserved: drift = {energy_stats['relative_drift']*100:.2f}%"


def test_numerical_stability():
    """Test numerical stability checking."""
    # Create stable simulation
    string = String(length=50.0, num_points=500, tension=50.0)
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=0.5)

    solver = VerletSolver(string, enable_force=False)
    solver.solve(total_time=1.0, dt=0.001, save_interval=50, verbose=False)

    stability_stats = analysis.check_numerical_stability(solver.displacement_history)

    assert stability_stats['is_stable']
    assert not stability_stats['has_nan']
    assert not stability_stats['has_inf']
    assert not stability_stats['has_explosion']


def test_energy_positive():
    """Test that energy values are positive."""
    string = String(length=50.0, num_points=500, tension=50.0)
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    solver = VerletSolver(string, enable_force=False)
    solver.solve(total_time=0.5, dt=0.001, save_interval=25, verbose=False)

    # All energy values should be non-negative
    for ke, pe, total in solver.energy_history:
        assert ke >= 0, "Kinetic energy should be non-negative"
        assert total >= 0, "Total energy should be non-negative"


def test_energy_exchange():
    """Test energy exchange between kinetic and potential."""
    string = String(length=50.0, num_points=500, tension=50.0)
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    solver = VerletSolver(string, enable_force=False)
    # Use even smaller dt for this test
    solver.solve(total_time=1.0, dt=0.0003, save_interval=100, verbose=False)

    ke_values = [e[0] for e in solver.energy_history]
    pe_values = [e[1] for e in solver.energy_history]

    # Initial state: mostly potential energy
    assert pe_values[0] > ke_values[0]

    # Energy should oscillate between kinetic and potential
    # Check that both KE and PE vary significantly
    ke_range = max(ke_values) - min(ke_values)
    pe_range = max(pe_values) - min(pe_values)

    assert ke_range > 0, "Kinetic energy should vary"
    assert pe_range > 0, "Potential energy should vary"


def test_energy_with_damping():
    """Test that energy decreases with damping."""
    string = String(length=50.0, num_points=500, tension=50.0)
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    solver = VerletSolver(string, enable_force=False, damping=0.1)
    # Use smaller dt
    solver.solve(total_time=1.0, dt=0.0003, save_interval=100, verbose=False)

    total_energies = [e[2] for e in solver.energy_history]

    # With damping, total energy should decrease
    assert total_energies[-1] < total_energies[0]

    # Energy should monotonically decrease (or stay constant)
    for i in range(1, len(total_energies)):
        assert total_energies[i] <= total_energies[i-1] * 1.01  # Allow small numerical fluctuations


def test_whip_energy_tapered():
    """Test energy conservation in whip simulation with tapered density."""
    whip = String(
        length=50.0,
        num_points=500,
        tension=100.0,
        density_profile='tapered',
        density_base=0.02,
        density_tip=0.002,
        taper_exponent=1.5
    )
    whip.set_initial_pulse(position=10.0, amplitude=1.0, width=2.0)

    solver = VerletSolver(whip, enable_force=False, damping=0.0)

    # Use very small timestep for stability with tapered density
    solver.solve(total_time=0.5, dt=0.0002, save_interval=100, verbose=False)

    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history,
        solver.energy_history,
        tolerance=0.15,  # More lenient for complex simulation
        plot=False
    )

    # Should still conserve energy reasonably well
    assert energy_stats['relative_drift'] < 0.20, f"Excessive energy drift: {energy_stats['relative_drift']*100:.2f}%"
