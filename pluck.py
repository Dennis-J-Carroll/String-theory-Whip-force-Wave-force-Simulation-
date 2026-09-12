"""
Click-to-pluck — turn pointer input into initial conditions.

Every user's first instinct in a wave simulator is to *poke the string*.
Until now the dashboard made them formalize that instinct as slider values
first. This module is the direct-manipulation path: a click on the wave
chart becomes a Gaussian pluck centered at the clicked position, and a
drag-drawn shape can be merged over the rest state.

The physics stays honest by construction: a pluck is just another initial
condition, so it re-enters the exact same solver path as the RUN button —
the CFL badge, escape readout, and conserved energy ledger all keep
working with no special cases.

Usage:
    from pluck import pluck_displacement, draw_displacement

    u0, u0_prev = pluck_displacement(x, click_x, amplitude, width)
    u_history, t, v = wave_solver(x, u0, u0_prev, c, dt, T, K1, K2,
                                  return_velocity=True)

Displacements returned here are PERTURBATIONS about the rest state (the
solver adds the well equilibrium u* internally — see ``solver.wave_solver``).
"""

import numpy as np

from solver import WELL_U_STAR

__all__ = ["pluck_displacement", "draw_displacement"]


def pluck_displacement(x, click_x, amplitude, width, *, u_star=WELL_U_STAR):
    """A rest-state string with a Gaussian pluck centered at ``click_x``.

    Args:
        x: spatial grid (1D, evenly spaced).
        click_x: where the finger landed (m). Clipped into the grid; the
            pluck peak sits at the nearest grid node.
        amplitude: peak displacement (m), i.e. the height above the rest
            state u*.
        width: Gaussian sigma (m).
        u_star: rest-state offset kept for signature completeness; the
            returned perturbation does NOT include it (the solver adds it).

    Returns:
        (u0, u0_prev): the perturbation profile and the step-before profile.
        A pluck starts from rest (v = 0), and the solver derives velocity
        as (u0 − u0_prev) / dt, so the step-before profile must EQUAL u0
        (zeros would inject a phantom v = u0/dt impulse).
    """
    x = np.asarray(x, dtype=float)
    amplitude = float(np.clip(amplitude, 0.0, None))
    width = float(width)
    if width <= 0:
        raise ValueError("pluck width must be positive")

    click_x = float(np.clip(click_x, x[0], x[-1]))
    center = x[int(np.argmin(np.abs(x - click_x)))]

    u0 = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    u0_prev = u0.copy()  # rest start: (u0 - u0_prev)/dt = 0
    return u0, u0_prev


def draw_displacement(x, drawn_u, *, blend=1.0):
    """Merge a drag-drawn shape over the rest state.

    The wave tab can offer freehand drawing: the pointer path is rasterized
    to the grid (zeros where the pointer never passed) and handed here.

    Args:
        x: spatial grid.
        drawn_u: drawn displacement per grid node (same shape as x).
        blend: 0..1 weight of the drawn shape over the rest state.

    Returns:
        (u0, u0_prev): merged perturbation profile and the rest-start
        step-before profile (equal to u0 — see pluck_displacement).
    """
    x = np.asarray(x, dtype=float)
    drawn = np.asarray(drawn_u, dtype=float)
    if drawn.shape != x.shape:
        raise ValueError(
            f"drawn shape {drawn.shape} does not match grid {x.shape}")
    u0 = float(blend) * drawn
    u0_prev = u0.copy()  # rest start: (u0 - u0_prev)/dt = 0
    return u0, u0_prev
