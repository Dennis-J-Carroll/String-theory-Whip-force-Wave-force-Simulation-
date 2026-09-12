"""Tests for the interactive wave-lab features: pointer pluck and regime
landscapes.

The pluck is "just another initial condition" — these tests prove it:
symmetric splitting, wavefronts at c, and a conserved energy ledger. The
landscapes are pinned to the same helpers the dashboard's live readout
uses, so the map and the pointer can never disagree.
"""

import numpy as np
import pytest

from solver import (
    WAVE_SCALE_K1,
    WAVE_SCALE_K2,
    WELL_U_STAR,
    wave_solver,
)
from pluck import pluck_displacement
from regimes import escape_landscape, stability_landscape
from interactive_visualization import (
    create_escape_landscape,
    create_stability_landscape,
)


class TestPluck:
    def test_peak_lands_at_click_and_clips_to_grid(self):
        x = np.arange(0, 50.1, 0.1)
        u0, _ = pluck_displacement(x, 12.3, 1.5, 2.0)
        assert x[np.argmax(u0)] == pytest.approx(12.3, abs=0.11)

        edge, _ = pluck_displacement(x, 999.0, 1.0, 2.0)  # far outside
        assert x[np.argmax(edge)] == pytest.approx(x[-1], abs=0.11)

    def test_rest_start_has_zero_velocity(self):
        x = np.arange(0, 50.1, 0.1)
        u0, u0_prev = pluck_displacement(x, 25.0, 1.2, 2.0)
        # The shim derives v0 = (u0 - u0_prev) / dt for any positive dt.
        assert np.allclose(u0, u0_prev)

    def test_pluck_splits_symmetrically_at_wave_speed(self):
        """With a negligible well, a rest pluck splits into two fronts at c.

        (At the calibrated well this does NOT happen — see the ringing
        test below: the dispersion relation is w^2 = c^2 k^2 + w0^2, and
        w0 = 5 rad/s pins a small pluck nearly in place. Scaling K1/K2
        down uniformly keeps u* = 2 but drives w0 -> 0.)
        """
        x = np.arange(0, 50.1, 0.1)
        c, dt, T = 1.0, 0.05, 4.0
        x0 = 25.0
        tiny = 1e-6  # scale both constants; u* = (2K1/K2)^(1/6) unchanged
        k1, k2 = WAVE_SCALE_K1 * tiny, WAVE_SCALE_K2 * tiny
        u0, u0_prev = pluck_displacement(x, x0, 0.8, 1.5)
        u_hist, t = wave_solver(x, u0, u0_prev, c, dt, T, k1, k2)

        idx = int(0.8 * len(t))
        u = u_hist[idx] - WELL_U_STAR
        elapsed = t[idx]

        # d'Alembert: the released pulse splits into two half-amplitude
        # fronts (always positive — no sign inversion). Split the energy
        # at the pluck point; each half's centroid travels at c.
        energy = u ** 2
        def centroid(mask):
            w = np.where(mask, energy, 0.0)
            return np.sum(x * w) / np.sum(w)
        right = centroid(x > x0)
        left = centroid(x < x0)

        assert right == pytest.approx(x0 + c * elapsed, abs=1.6)
        assert left == pytest.approx(x0 - c * elapsed, abs=1.6)

    def test_pluck_rings_in_the_calibrated_well(self):
        """At calibration, the well traps a small pluck: it rings, not flies.

        The dispersion relation w^2 = c^2 k^2 + w0^2 (w0 = 5 rad/s) gives
        a Gaussian of width 1.5 m a group velocity ~0.13 m/s, so after a
        plain-wave pulse would have crossed ±3 m the energy still sits at
        the pluck point — oscillating about u*. This is the simulator's
        whole premise: the wave force binds the wave.
        """
        x = np.arange(0, 50.1, 0.1)
        c, dt, T = 1.0, 0.05, 4.0
        x0 = 25.0
        u0, u0_prev = pluck_displacement(x, x0, 0.8, 1.5)
        u_hist, t = wave_solver(x, u0, u0_prev, c, dt, T,
                                WAVE_SCALE_K1, WAVE_SCALE_K2)

        idx = int(0.8 * len(t))
        u = u_hist[idx] - WELL_U_STAR
        elapsed = t[idx]
        energy = u ** 2
        center = np.sum(x * energy) / np.sum(energy)
        # A free string would have moved the lobes ~c*elapsed = 3.2 m;
        # the well keeps the energy centroid within one pluck width.
        assert abs(center - x0) < 1.5
        # And it is genuinely oscillating: significant amplitude survives.
        assert u.max() > 0.2 * 0.8

    def test_plucked_run_conserves_energy(self):
        """The pluck path must honor the same conserved ledger as RUN."""
        x = np.arange(0, 50.1, 0.1)
        c, dt, T = 1.0, 0.05, 6.0
        u0, u0_prev = pluck_displacement(x, 18.0, 1.0, 2.0)
        u_hist, t, v_hist = wave_solver(x, u0, u0_prev, c, dt, T,
                                        WAVE_SCALE_K1, WAVE_SCALE_K2,
                                        return_velocity=True)
        dx = x[1] - x[0]
        ke = 0.5 * np.mean(v_hist ** 2, axis=1)
        elastic = (0.5 * c ** 2 * np.diff(u_hist, axis=1) ** 2 / dx ** 2).mean(axis=1)
        from solver import potential_function
        well = (potential_function(u_hist, WAVE_SCALE_K1, WAVE_SCALE_K2)
                - potential_function(WELL_U_STAR, WAVE_SCALE_K1, WAVE_SCALE_K2)).mean(axis=1)
        total = ke + elastic + well
        drift = abs(total[-1] - total[0]) / max(abs(total[0]), 1e-12)
        assert drift < 0.05


class TestEscapeLandscape:
    def test_shape_and_axes(self):
        g = escape_landscape(n=11)
        assert g["ratio"].shape == (11, 11)
        assert len(g["axes"]["amplitude"]) == 11
        assert len(g["axes"]["width"]) == 11
        # ratio increases monotonically with amplitude along any column
        col = g["ratio"][:, 5]
        assert np.all(np.diff(col) > 0)

    def test_matches_single_point_readout(self):
        """The map must agree with the dashboard's exact helper."""
        from solver import crest_energy, well_properties
        g = escape_landscape(n=15, c=1.0)
        A = g["axes"]["amplitude"][7]
        W = g["axes"]["width"][4]
        depth = well_properties(WAVE_SCALE_K1, WAVE_SCALE_K2)["depth"]
        expected = crest_energy(A, W, c=1.0)["total"] / depth
        assert g["ratio"][7, 4] == pytest.approx(expected, rel=1e-12)

    def test_boundary_near_known_crossing(self):
        """A = 2, sigma = 2, c = 1 crosses ratio 1 (dashboard math)."""
        g = escape_landscape(n=21, c=1.0)
        A = g["axes"]["amplitude"]
        W = g["axes"]["width"]
        iA = int(np.argmin(np.abs(A - 2.0)))
        iW = int(np.argmin(np.abs(W - 2.0)))
        assert g["ratio"][iA, iW] == pytest.approx(1.0, rel=0.25)


class TestStabilityLandscape:
    def test_cfl_boundary_lights_up(self):
        """Drift beyond CFL 1 should exceed drift well below it."""
        g = stability_landscape(n=5, total_time=3.0, length=20.0)
        dtf = g["axes"]["dt_frac"]
        i_stable = int(np.argmin(np.abs(dtf - 0.4)))
        i_unstable = int(np.argmax(dtf > 1.2))
        stable_drift = np.median(g["drift"][i_stable, :])
        unstable_drift = np.median(g["drift"][i_unstable, :])
        assert unstable_drift > 10 * stable_drift

    def test_shape_and_axes(self):
        g = stability_landscape(n=3, total_time=1.0, length=15.0)
        assert g["drift"].shape == (3, 3)
        assert np.all(np.isfinite(g["drift"]))


class TestLandscapeFigures:
    def test_escape_figure_builds(self):
        g = escape_landscape(n=9)
        fig = create_escape_landscape(g["ratio"], g["axes"]["amplitude"],
                                      g["axes"]["width"], g["depth"])
        assert len(fig.data) == 2  # heatmap + ratio-1 contour
        assert fig.data[0].type == "heatmap"

    def test_stability_figure_builds(self):
        g = stability_landscape(n=3, total_time=1.0, length=15.0)
        fig = create_stability_landscape(g["drift"], g["axes"]["dt_frac"],
                                         g["axes"]["dx"])
        assert fig.data[0].type == "heatmap"
        assert np.allclose(fig.data[0].z,
                           np.log10(np.maximum(g["drift"], 1e-300)))
