"""
Tests for the Accuracy Lab: convergence orders and the exact-solution harness.

The lab's entire value is that its slopes tell the truth, so the tests pin the
measured convergence orders:

- spatial study: every integrator converges as ~dx^2 (the grid sets spatial
  error, not the time scheme)
- temporal study: CD and Verlet (same stencil) drift identically at ~dt^2,
  while RK4 falls far faster (its leading error on linear waves is pure
  amplitude damping, ~dt^5)
- d'Alembert invariants: the exact solution stays fixed-ended and even in
  time (velocity release: u(0) = g(x))
"""

import math

import numpy as np
import pytest

import accuracy_lab as lab
from solver import CentralDifferenceSolver, RK4Solver, VerletSolver


# ----------------------------------------------------------------------------
# d'Alembert exact-solution invariants
# ----------------------------------------------------------------------------

def test_dalembert_zero_velocity_at_t0():
    """Free release: u_t(x,0)=0, so u(eps)-u(0) scales as eps^2 (acceleration
    is finite), NOT eps. A linear-in-eps drift would mean a non-zero initial
    velocity — exactly the bug class this lab exists to catch."""
    x = np.linspace(0.0, lab.LENGTH, 801)
    c = math.sqrt(lab.TENSION / lab.DENSITY)
    u0 = lab.exact_dalembert(x, 0.0, c, lab.LENGTH)
    eps = 1e-3
    d1 = np.max(np.abs(lab.exact_dalembert(x, eps, c, lab.LENGTH) - u0))
    d2 = np.max(np.abs(lab.exact_dalembert(x, 2 * eps, c, lab.LENGTH) - u0))
    ratio = d2 / d1
    assert 3.0 < ratio < 5.0, f"u(eps)-u(0) ratio {ratio:.2f}, expected ~4 (eps^2)"


def test_dalembert_matches_initial_shape():
    """At t=0 the odd 2L-periodic extension must collapse to g(x) on [0, L]."""
    x = np.linspace(0.0, lab.LENGTH, 801)
    u0 = lab.exact_dalembert(x, 0.0, math.sqrt(lab.TENSION / lab.DENSITY),
                             lab.LENGTH)
    assert np.allclose(u0, lab._gaussian(x), atol=1e-12)


def test_dalembert_fixed_ends():
    """Fixed ends: u(0, t) = u(L, t) = 0 for all t (odd extension guarantees it)."""
    c = math.sqrt(lab.TENSION / lab.DENSITY)
    for time in (0.0, 0.17, 0.5, 1.3):
        x_ends = np.array([0.0, lab.LENGTH])
        u = lab.exact_dalembert(x_ends, time, c, lab.LENGTH)
        assert np.allclose(u, 0.0, atol=1e-12)


# ----------------------------------------------------------------------------
# Convergence studies — the measured orders are the contract
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spatial_results():
    return lab.run_spatial_study(mults=(1, 2, 4), total_time=0.5)


@pytest.fixture(scope="module")
def temporal_results():
    return lab.run_temporal_study(cfls=(0.8, 0.4, 0.2), total_time=0.5)


def test_spatial_all_integrators_second_order(spatial_results):
    """L2 error falls as dx^2 for every scheme — space, not time, sets it."""
    for name, pts in spatial_results.items():
        orders = lab.measured_order(pts, "l2")
        for order in orders:
            assert order == pytest.approx(2.0, abs=0.15), (
                f"{name}: measured spatial order {order:.2f}, expected ~2"
            )


def test_temporal_cd_and_verlet_same_stencil_dt2(temporal_results):
    """CD and Verlet share the stencil: identical drift, both ~dt^2."""
    cd_orders = lab.measured_order(temporal_results["Central difference"], "drift")
    vl_orders = lab.measured_order(temporal_results["Velocity Verlet"], "drift")
    for o in cd_orders + vl_orders:
        assert o == pytest.approx(2.0, abs=0.15), f"expected ~2, got {o:.2f}"
    cd_drifts = [p["drift"] for p in temporal_results["Central difference"]]
    vl_drifts = [p["drift"] for p in temporal_results["Velocity Verlet"]]
    assert cd_drifts == pytest.approx(vl_drifts, rel=1e-6)


def test_temporal_rk4_far_steeper_than_verlet(temporal_results):
    """RK4's drift must fall strictly faster than Verlet's dt^2."""
    rk4_orders = lab.measured_order(temporal_results["RK4"], "drift")
    for o in rk4_orders:
        assert o > 3.5, f"RK4 measured order {o:.2f}, expected > 3.5"


def test_rk4_drift_lowers_below_verlet_by_orders_of_magnitude(temporal_results):
    """At the finest dt, RK4 should be far below Verlet, not merely below."""
    rk4 = temporal_results["RK4"][-1]["drift"]
    vl = temporal_results["Velocity Verlet"][-1]["drift"]
    assert rk4 < vl / 50.0


def test_discrete_stencil_energy_matches_continuum_to_grid_error():
    """The two energy definitions agree up to the known O(dx^2) offset."""
    s = lab._new_string(401)
    c = math.sqrt(lab.TENSION / lab.DENSITY)
    u = lab.exact_dalembert(s.x, 0.2, c, lab.LENGTH)
    g = np.gradient(u, s.dx)
    v = -c * g  # d'Alembert right-going part: u_t = -c u_x (left-going cancels on average)
    e_disc = lab.discrete_stencil_energy(u, v, s.dx)
    assert e_disc > 0.0
    # coarse vs fine grid: the discrepancy must SHRINK as the grid refines
    s2 = lab._new_string(801)
    u2 = lab.exact_dalembert(s2.x, 0.2, c, lab.LENGTH)
    g2 = np.gradient(u2, s2.dx)
    e_disc2 = lab.discrete_stencil_energy(u2, -c * g2, s2.dx)
    # continuum reference
    x_fine = np.linspace(0.0, lab.LENGTH, 8001)
    u_f = lab.exact_dalembert(x_fine, 0.2, c, lab.LENGTH)
    e_cont = 0.5 * lab.TENSION * np.trapz(
        np.gradient(u_f, lab.LENGTH / 8000) ** 2, x_fine
    ) + 0.5 * lab.DENSITY * np.trapz((-c * np.gradient(u_f, lab.LENGTH / 8000)) ** 2, x_fine)
    err_coarse = abs(e_disc - e_cont) / e_cont
    err_fine = abs(e_disc2 - e_cont) / e_cont
    assert err_fine < err_coarse * 0.5
