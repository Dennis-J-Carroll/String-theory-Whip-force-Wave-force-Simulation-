"""Escape-analysis embedding in the interactive HTML report.

The report must carry the same numbers as the dashboard's PHYSICS card:
well depth (escape threshold) vs the crest energy of the initial Gaussian
pulse, with the green/amber/red thresholds decided in JS but the numbers
and the ratio pinned here on the Python side.
"""
import numpy as np
import pytest

from report import capture_run
from solver import crest_energy, well_properties

K1, K2 = 5689.6, 177.8          # the calibrated wave-scale pair (main.py)
C = 70.71                       # sqrt(T/mu) for the standard string

# Minimal synthetic run: u* = 2 background + Gaussian pulse, 60 frames.
X = np.linspace(0.0, 50.0, 200)
U0 = 2.0 + 0.5 * np.exp(-((X - 25.0) ** 2) / (2 * 5.0 ** 2))
FRAMES = [U0 * (1.0 - 0.01 * f) for f in range(60)]
TIMES = np.linspace(0.0, 3.0, 60)
ENERGY = [(1.0, 2.0, 3.0)] * 60  # (ke, pe, tot) triples


def _capture(**kw):
    base = dict(label="test", x=X, time_history=TIMES, frames=FRAMES)
    base.update(kw)
    return capture_run(**base)


def test_no_well_by_default():
    """Force-free runs (e.g. report.py's own demo) embed no escape analysis."""
    data = _capture()
    assert "well" not in data


def test_well_without_pulse_embeds_depth_and_turning_point():
    data = _capture(k1=K1, k2=K2)
    props = well_properties(K1, K2)
    assert data["well"]["u_star"] == pytest.approx(props["u_star"], abs=1e-4)
    assert data["well"]["depth"] == pytest.approx(props["depth"], rel=1e-9)
    assert data["well"]["turning_point"] == pytest.approx(props["turning_point"], abs=1e-4)
    assert "crest" not in data["well"]


def test_crest_embedded_with_pulse_geometry():
    data = _capture(k1=K1, k2=K2, pulse_amplitude=0.5, pulse_width=5.0, pulse_speed=C)
    crest = crest_energy(0.5, 5.0, c=C, k1=K1, k2=K2)
    w = data["well"]["crest"]
    assert w["well"] == pytest.approx(crest["well"], abs=1e-6)
    assert w["elastic"] == pytest.approx(crest["elastic"], abs=1e-6)
    assert w["total"] == pytest.approx(crest["total"], abs=1e-6)
    assert w["ratio"] == pytest.approx(crest["total"] / well_properties(K1, K2)["depth"])


def test_analytic_depth_and_geometry():
    """K1/K2 = 32 exactly, so u* = 64^(1/6) = 2 m and the V = 0 crossing
    sits at 32^(1/6) = 2^(5/6) — both pinned against closed forms."""
    data = _capture(k1=K1, k2=K2)
    assert data["well"]["u_star"] == pytest.approx(2.0, rel=1e-9)
    assert data["well"]["depth"] == pytest.approx(1.3890625, rel=1e-9)
    assert data["well"]["turning_point"] == pytest.approx(2.0 ** (5.0 / 6.0), abs=5e-5)


def test_red_state_numbers():
    """A = 2 must be past 100% — the wall-slam regime (Mission 2 territory)."""
    data = _capture(k1=K1, k2=K2, pulse_amplitude=2.0, pulse_width=5.0, pulse_speed=C)
    assert data["well"]["crest"]["ratio"] >= 1.0


def test_default_main_py_pulse_is_deep_red():
    """main.py's run (A=0.5, width=5, c=70.71) is far past 100%: the elastic
    term ½c²(du/dx)² dominates at real wave speed. Probed reality: nodes of
    that very run visit u = −0.50, well past the V = 0 crossing — so the
    red verdict the report will show is honest, not an artifact."""
    data = _capture(k1=K1, k2=K2, pulse_amplitude=0.5, pulse_width=5.0, pulse_speed=C)
    cr = data["well"]["crest"]
    assert cr["well"] == pytest.approx(0.75605, rel=1e-3)     # V(2.5) − V(2)
    assert cr["elastic"] == pytest.approx(9.197, rel=1e-3)    # ½c²·(A·e^−½/σ)²
    assert cr["total"] == pytest.approx(9.9531, rel=1e-3)
    assert cr["ratio"] > 5.0


def test_dashboard_scenario_is_amber():
    """The dashboard's exact scenario (A=1, width=2, c=1) sits at ~87% —
    the amber 'near escape' band shared by both surfaces."""
    data = _capture(k1=K1, k2=K2, pulse_amplitude=1.0, pulse_width=2.0, pulse_speed=1.0)
    ratio = data["well"]["crest"]["ratio"]
    assert 0.85 <= ratio < 1.0
    assert ratio == pytest.approx(0.865, rel=2e-2)


def test_energy_and_cfl_still_embedded():
    data = _capture(energy_history=ENERGY, cfl=0.71, k1=K1, k2=K2,
                    pulse_amplitude=0.5, pulse_width=5.0)
    assert "energy" in data and "cfl" in data and "well" in data
    assert data["cfl"] == 0.71


def test_turning_point_below_equilibrium():
    """The V = 0 crossing must sit below u* (it is u*/2^(1/6))."""
    data = _capture(k1=K1, k2=K2)
    assert data["well"]["turning_point"] < data["well"]["u_star"]
