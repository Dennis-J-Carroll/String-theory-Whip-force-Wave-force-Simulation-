"""Tests for the well-geometry and crest-energy helpers.

These pin the physics behind the dashboard's well-depth / escape-energy
readout: the equilibrium placement, the depth as escape threshold, the
V = 0 turning point, the scaling laws, and the bound/unbound boundary the
amplitude warning is calibrated to.
"""

import numpy as np
import pytest

from solver import (
    WAVE_SCALE_K1,
    WAVE_SCALE_K2,
    WELL_U_STAR,
    crest_energy,
    force_function,
    potential_function,
    well_properties,
)


class TestWellProperties:
    def test_equilibrium_is_calibrated_u_star(self):
        props = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)
        assert props["u_star"] == pytest.approx(WELL_U_STAR, rel=1e-12)

    def test_force_vanishes_at_equilibrium(self):
        props = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)
        assert force_function(props["u_star"], WAVE_SCALE_K1, WAVE_SCALE_K2) \
            == pytest.approx(0.0, abs=1e-9)

    def test_depth_is_negated_minimum(self):
        props = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)
        v_min = potential_function(props["u_star"], WAVE_SCALE_K1, WAVE_SCALE_K2)
        assert v_min < 0
        assert props["depth"] == pytest.approx(-v_min, rel=1e-12)

    def test_potential_zero_at_turning_point(self):
        props = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)
        assert potential_function(props["turning_point"], WAVE_SCALE_K1,
                                  WAVE_SCALE_K2) == pytest.approx(0.0, rel=1e-9)

    def test_uniform_scaling_moves_depth_not_equilibrium(self):
        # k1, k2 -> s*k1, s*k2: the ratio k1/k2 is unchanged, so u* stays
        # put and every energy (depth, crest well part) scales by s.
        props = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)
        scaled = well_properties(3.0 * WAVE_SCALE_K1, 3.0 * WAVE_SCALE_K2)
        assert scaled["u_star"] == pytest.approx(props["u_star"], rel=1e-12)
        assert scaled["depth"] == pytest.approx(3.0 * props["depth"], rel=1e-12)

    def test_rejects_nonpositive_constants(self):
        with pytest.raises(ValueError):
            well_properties(0.0, 1.0)
        with pytest.raises(ValueError):
            well_properties(1.0, -1.0)


class TestCrestEnergy:
    def test_zero_amplitude_carries_nothing(self):
        e = crest_energy(0.0, 2.0, c=1.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        assert e["well"] == pytest.approx(0.0, abs=1e-12)
        assert e["elastic"] == pytest.approx(0.0, abs=1e-12)
        assert e["total"] == pytest.approx(0.0, abs=1e-12)

    def test_elastic_matches_gaussian_max_gradient(self):
        # Independent check: the steepest slope of A*exp(-(x)^2/(2w^2)) is
        # A*e^-0.5/w — verify against a numeric gradient of the actual pulse.
        A, w = 1.7, 2.4
        e = crest_energy(A, w, c=1.3, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        x = np.linspace(-20, 20, 200001)
        g = np.abs(np.gradient(A * np.exp(-(x**2) / (2 * w**2)), x))
        assert e["elastic"] == pytest.approx(0.5 * 1.3**2 * g.max()**2, rel=1e-3)

    def test_total_increases_with_amplitude(self):
        totals = [crest_energy(A, 2.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)["total"]
                  for A in (0.5, 1.0, 2.0, 4.0)]
        assert totals == sorted(totals)

    def test_default_pulse_is_bound_but_close_to_escape(self):
        # The dashboard's default (A = 1 m, width = 2 m) must sit under the
        # escape threshold — but near it, which is what makes the readout
        # worth watching.
        e = crest_energy(1.0, 2.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        assert 0.5 < e["total"] / e["depth"] < 1.0

    def test_doubling_amplitude_crosses_escape(self):
        # A = 2 m at width 2 m: crest energy above well depth — the amber
        # warning state the dashboard shows.
        e = crest_energy(2.0, 2.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        assert e["total"] > e["depth"]

    def test_narrower_width_adds_elastic_energy(self):
        wide = crest_energy(1.0, 4.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        narrow = crest_energy(1.0, 1.0, k1=WAVE_SCALE_K1, k2=WAVE_SCALE_K2)
        assert narrow["elastic"] > wide["elastic"]
        # The well part depends on amplitude only, not width.
        assert narrow["well"] == pytest.approx(wide["well"], rel=1e-12)
