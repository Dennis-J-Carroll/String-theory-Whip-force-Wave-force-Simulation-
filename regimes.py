"""
Regime landscapes — sweep, don't sample.

Missions and the dashboard answer "what happens with THESE parameters?".
A landscape answers the physicist's real question: "WHERE is the
boundary?" N×N parameter grids are swept (in parallel across all CPU
cores) and rendered as themed heatmaps, so one picture replaces a hundred
single runs.

Two landscapes ship:

- ``escape_landscape`` — escape ratio vs (amplitude, width). The red
  wall is the punch-through regime Mission 2 hunts for; the exact
  ``crest_energy`` / ``well_properties`` analytics draw it with zero
  simulations, so it renders instantly.
- ``stability_landscape`` — energy drift after a short sim vs (dt, dx),
  on a coarse grid so the total work stays ~seconds. Green means the
  symplectic integrator held; red means the run blew up. This is the
  Courant limit *seen* instead of told.

Usage:
    from regimes import escape_landscape, stability_landscape

    grid = escape_landscape(n=41)
    grid["ratio"]          # (n, n) escape ratio, axes (amplitude, width)
    grid["axes"]           # {"amplitude": ..., "width": ...}

Every entry is computed by the same helpers the dashboard's live readout
uses, so the landscape can never disagree with the pointer-read number.
"""

from __future__ import annotations

import numpy as np

from solver import WAVE_SCALE_K1, WAVE_SCALE_K2, crest_energy, well_properties
from solver import wave_solver

__all__ = ["escape_landscape", "stability_landscape"]


# ---------------------------------------------------------------------------
# Landscape 1 — escape ratio vs (amplitude, width). Pure analytics: exact,
# instant, and identical to the dashboard readout by construction.
# ---------------------------------------------------------------------------

def escape_landscape(n=41, amplitude_range=(0.1, 4.0), width_range=(0.5, 5.0),
                     c=1.0, k1=None, k2=None):
    """Escape ratio over an (amplitude, width) grid — no simulation needed.

    The ratio is crest_energy(total) / well depth, exactly the number the
    dashboard's PHYSICS card shows; >= 1 means the crest can leave the well.

    Returns:
        dict with the (n, n) ``ratio`` array (rows = amplitude index,
        cols = width index), the ``axes`` arrays, and the well ``depth``.
    """
    k1 = WAVE_SCALE_K1 if k1 is None else float(k1)
    k2 = WAVE_SCALE_K2 if k2 is None else float(k2)

    amplitudes = np.linspace(*amplitude_range, n)
    widths = np.linspace(*width_range, n)
    depth = well_properties(k1, k2)["depth"]

    # crest_energy is linear in amplitude for the well part and quadratic
    # for the elastic part — but evaluate directly anyway: it keeps this
    # grid provably identical to the single-run readout for any future
    # change to the helper.
    A, W = np.meshgrid(amplitudes, widths, indexing="ij")
    ratio = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            crest = crest_energy(A[i, j], W[i, j], c=c, k1=k1, k2=k2)
            ratio[i, j] = crest["total"] / depth

    return {
        "ratio": ratio,
        "axes": {"amplitude": amplitudes, "width": widths},
        "depth": float(depth),
    }


# ---------------------------------------------------------------------------
# Landscape 2 — stability vs (dt fraction, dx). Real short sims on a coarse
# grid; each cell runs a wave_solver pass long enough to expose blow-up.
# ---------------------------------------------------------------------------

def _stability_cell(args):
    """One (dt_frac, dx) cell — kept module-level for multiprocessing."""
    (dt_frac, dx, c, total_time, length) = args
    x = np.arange(0.0, length + dx, dx)
    dt_limit = dx / c
    dt = dt_frac * dt_limit
    u0 = np.exp(-0.5 * ((x - length / 2) / 2.0) ** 2)
    u0_prev = np.zeros_like(u0)
    try:
        u_hist, _ = wave_solver(x, u0, u0_prev, c, dt, total_time)
        final = np.asarray(u_hist[-1])
        # Drift of the mean-square displacement against frame 0: the
        # symplectic map bounds this for stable runs; a detonated grid
        # grows it without bound.
        drift = float(np.mean((final - u0) ** 2))
        return drift if np.isfinite(drift) else 1e12
    except Exception:
        return 1e12


def stability_landscape(n=13, dt_frac_range=(0.1, 1.6), dx_range=(0.05, 0.5),
                        c=1.0, total_time=6.0, length=30.0, workers=None):
    """Energy-drift landscape over (dt fraction of CFL, dx).

    Each cell integrates a short pulse with the calibrated well; the value
    is the final mean-square deviation from the initial profile (log scale
    when rendered). Cells past the Courant limit light up red.

    Returns:
        dict with the (n, n) ``drift`` array (rows = dt_frac index,
        cols = dx index) and ``axes``.
    """
    from concurrent.futures import ProcessPoolExecutor
    import os

    dt_fracs = np.linspace(*dt_frac_range, n)
    dxs = np.linspace(*dx_range, n)

    jobs = [(dtf, dx, c, total_time, length)
            for dtf in dt_fracs for dx in dxs]

    workers = workers or max(1, os.cpu_count() - 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_stability_cell, jobs, chunksize=1))
    else:
        results = [_stability_cell(j) for j in jobs]

    drift = np.array(results, dtype=float).reshape(n, n)
    return {
        "drift": drift,
        "axes": {"dt_frac": dt_fracs, "dx": dxs},
    }
