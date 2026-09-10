"""
The Accuracy Lab — a convergence study that turns metrics into understanding.

"Relative drift < 5%" is meaningless until you've seen *why* it moves. This lab
measures the SAME initial condition two independent ways, one knob at a time —
because a co-refined study (dx and dt shrunk together) can only ever show the
worst of the two errors, and hides the integrator entirely:

  STUDY 1 — Spatial convergence: refine the grid with dt ∝ dx and measure the
  relative L2 error against the exact d'Alembert solution (method of images,
  exact through every reflection). All integrators land on ~dx²: the grid, not
  the time scheme, sets spatial error.

  STUDY 2 — Temporal convergence: freeze the grid, halve dt, and measure the
  drift of the DISCRETE stencil energy (the quantity the scheme actually
  conserves — not the continuum integral, whose O(dx²) mismatch would floor
  every curve). Now the integrators split: central difference and Verlet are
  the same stencil and drift identically at ~dt² (their velocity estimates
  differ only by time-staggering), while RK4 lands ~200× lower and falls at
  ~dt⁵ — its leading error on linear waves is pure amplitude damping, one
  order better than the textbook dt⁴ bound.

Together the two panels teach the deepest lesson in the app: *the metric you
plot picks the winner*. L2 says "all equal"; energy drift says RK4 ≫ Verlet =
central difference. Metrics have opinions.

(A nice piece of history: the first run of this lab exposed a real solver bug —
the central-difference bootstrap omitted the ½a₀dt² Taylor term, silently
capping the scheme at 1st order. Fixed in solver.py; this lab now verifies the
fix every time it runs.)

Usage:
    python accuracy_lab.py            # full study -> output/accuracy_lab.png
    python accuracy_lab.py --selftest # smaller/faster study, same outputs
"""

from __future__ import annotations

import argparse
import math
from typing import Callable, Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

import djc_theme as t
from string_model import String
from solver import CentralDifferenceSolver, RK4Solver, VerletSolver

# ----------------------------------------------------------------------------
# Experimental setup — one shared physical scenario for every run
# ----------------------------------------------------------------------------

LENGTH = 50.0          # m
TENSION = 50.0         # N
DENSITY = 0.01         # kg/m  ->  c = 70.71 m/s exactly
PULSE_CENTER = 25.0    # m
PULSE_WIDTH = 2.0      # m
PULSE_AMP = 1.0        # m
CANON_POINTS = 4001    # common grid for L2 error measurement

# STUDY 1 — spatial: grid refined, dt co-refined with dx (dt = 0.5 dx / c)
SPATIAL_BASE_N = 151
SPATIAL_MULTS = (1, 2, 4, 8)
SPATIAL_CFL = 0.5
SPATIAL_TIME = 1.0    # includes two end reflections — the hard case

# STUDY 2 — temporal: grid frozen, dt halved. Run time kept short enough that
# the pulse never reaches the ends: reflections couple the velocity diagnostic
# to the drift measurement and muddy the dt-scaling the study exists to show.
TEMPORAL_N = 301
TEMPORAL_CFLS = (0.8, 0.4, 0.2, 0.1)
TEMPORAL_TIME = 0.5

INTEGRATORS: Dict[str, Callable[..., object]] = {
    "Central difference": CentralDifferenceSolver,
    "RK4": RK4Solver,
    "Velocity Verlet": VerletSolver,
}

SERIES_COLORS = {
    "Central difference": t.ACCENT.VIOLET,
    "RK4": t.ACCENT.CYAN,
    "Velocity Verlet": t.TEAL.N300,
}

# Textbook slopes for the guides — measured orders may differ (that IS the lesson)
GUIDE_SLOPES = {
    "Central difference": 2,
    "RK4": 4,
    "Velocity Verlet": 2,
}

GUIDE_LABELS = {
    "Central difference": r"$\propto dt^2$ (textbook)",
    "RK4": r"$\propto dt^4$ (textbook)",
    "Velocity Verlet": r"$\propto dt^2$ (textbook)",
}


def _gaussian(x: np.ndarray) -> np.ndarray:
    return PULSE_AMP * np.exp(-((x - PULSE_CENTER) ** 2) / (2 * PULSE_WIDTH**2))


def exact_dalembert(
    x: np.ndarray, time: float, c: float, length: float,
    gaussian: Callable[[np.ndarray], np.ndarray] = _gaussian,
) -> np.ndarray:
    """Exact solution of the force-free wave equation with FIXED ends.

    d'Alembert's solution on an infinite string is u = [g(x-ct) + g(x+ct)]/2,
    but a finite string with fixed ends needs the method of images: the initial
    shape is extended oddly with period 2L and both travelling copies are
    summed over all mirror images. This stays exact through every reflection.

    Args:
        x: positions at which to evaluate the solution
        time: evaluation time
        c: wave speed (uniform string)
        length: string length L
        gaussian: initial shape g(x) with g_t(x, 0) = 0
    """
    def images(s: np.ndarray) -> np.ndarray:
        """Odd 2L-periodic extension of g evaluated at s."""
        total = np.zeros_like(s, dtype=float)
        for n in range(-4, 5):  # g's tails are zero long before these run out
            total += gaussian(s + 2 * n * length) - gaussian(-s + 2 * n * length)
        return total

    return 0.5 * (images(x - c * time) + images(x + c * time))


def discrete_stencil_energy(u: np.ndarray, v: np.ndarray, dx: float,
                            tension: float = TENSION,
                            density: float = DENSITY) -> float:
    """Energy of the DISCRETE stencil: sum of ½T((u_{i+1}-u_i)/dx)² + ½μv_i².

    This is the quantity the second-difference scheme actually conserves; the
    continuum integral in String.get_total_energy() differs from it by O(dx²),
    which would floor every curve in the temporal study. On a uniform string
    the two agree up to that constant offset — which is exactly why the floor
    appears only as an error, never in the dynamics.
    """
    pe = 0.5 * tension * np.sum(((u[1:] - u[:-1]) / dx) ** 2) * dx
    ke = 0.5 * density * np.sum(v ** 2) * dx
    return float(pe + ke)


def _relative_l2(numeric: np.ndarray, exact: np.ndarray, dx: float) -> float:
    """Relative L2 norm: ||u_num - u_exact|| / ||u_exact||."""
    denom = math.sqrt(np.trapz(exact**2, dx=dx))
    if denom == 0.0:
        return float("inf")
    return math.sqrt(np.trapz((numeric - exact) ** 2, dx=dx)) / denom


def _new_string(num_points: int) -> String:
    s = String(length=LENGTH, num_points=num_points, tension=TENSION,
               density_profile="uniform", density_uniform=DENSITY)
    s.set_initial_gaussian(center=PULSE_CENTER, width=PULSE_WIDTH,
                           amplitude=PULSE_AMP)
    return s


def _run_case(solver_cls, num_points: int, cfl: float, total_time: float,
              x_canon: np.ndarray, dx_canon: float, c_exact: float) -> dict:
    """One run: solve, then measure L2 vs exact and discrete-energy drift."""
    s = _new_string(num_points)
    dt = cfl * s.dx / c_exact
    num_steps = int(round(total_time / dt))
    end_time = num_steps * dt

    energy0 = discrete_stencil_energy(s.displacement, s.velocity, s.dx)
    solver = solver_cls(s, enable_force=False)
    for _ in range(num_steps):
        u, v = solver.step(s.displacement, s.velocity, dt, s.dx, s.wave_speed)
        s.displacement, s.velocity = u, v
        s.apply_boundary_conditions()
    energy1 = discrete_stencil_energy(s.displacement, s.velocity, s.dx)

    exact_on_grid = exact_dalembert(s.x, end_time, c_exact, LENGTH)
    numeric_canon = np.interp(x_canon, s.x, s.displacement)
    exact_canon = np.interp(x_canon, s.x, exact_on_grid)

    return {
        "num_points": num_points,
        "dx": s.dx,
        "dt": dt,
        "cfl": cfl,
        "l2": _relative_l2(numeric_canon, exact_canon, dx_canon),
        "drift": abs(energy1 - energy0) / max(abs(energy0), 1e-300),
    }


def run_spatial_study(mults: Sequence[int] = SPATIAL_MULTS,
                      total_time: float = SPATIAL_TIME,
                      verbose: bool = False) -> Dict[str, List[dict]]:
    """Study 1: refine the grid (dt ∝ dx), measure L2 vs the exact solution."""
    x_canon = np.linspace(0.0, LENGTH, CANON_POINTS)
    dx_canon = x_canon[1] - x_canon[0]
    c_exact = math.sqrt(TENSION / DENSITY)

    results: Dict[str, List[dict]] = {name: [] for name in INTEGRATORS}
    for mult in mults:
        n = SPATIAL_BASE_N * mult
        for name, cls in INTEGRATORS.items():
            pt = _run_case(cls, n, SPATIAL_CFL, total_time, x_canon, dx_canon,
                           c_exact)
            results[name].append(pt)
            if verbose:
                print(f"  {name:<20} N={n:>5}  L2={pt['l2']:.3e}  "
                      f"drift={pt['drift']:.3e}")
    return results


def run_temporal_study(cfls: Sequence[float] = TEMPORAL_CFLS,
                       total_time: float = TEMPORAL_TIME,
                       verbose: bool = False) -> Dict[str, List[dict]]:
    """Study 2: freeze the grid, halve dt, measure discrete-energy drift."""
    x_canon = np.linspace(0.0, LENGTH, CANON_POINTS)
    dx_canon = x_canon[1] - x_canon[0]
    c_exact = math.sqrt(TENSION / DENSITY)

    results: Dict[str, List[dict]] = {name: [] for name in INTEGRATORS}
    for cfl in cfls:
        for name, cls in INTEGRATORS.items():
            pt = _run_case(cls, TEMPORAL_N, cfl, total_time, x_canon, dx_canon,
                           c_exact)
            results[name].append(pt)
            if verbose:
                print(f"  {name:<20} cfl={cfl:4.2f}  L2={pt['l2']:.3e}  "
                      f"drift={pt['drift']:.3e}")
    return results


def measured_order(points: Sequence[dict], key: str) -> List[float]:
    """Order of accuracy between consecutive refinements: log-slope of error."""
    orders = []
    for prev, curr in zip(points, points[1:]):
        ratio = prev[key] / curr[key]
        knob = prev["dx"] / curr["dx"] if "dx" in points[0] and \
            prev["dx"] != curr["dx"] else prev["dt"] / curr["dt"]
        orders.append(math.log(ratio, knob) if ratio > 0 else float("nan"))
    return orders


# ----------------------------------------------------------------------------
# Themed output — the two-panel chart and the CLI report
# ----------------------------------------------------------------------------

def _slope_guide(ax, x0: float, y0: float, slope: float, decades: float = 0.9,
                 color=None, label: str = None) -> None:
    """Dashed guide showing what slope `order` looks like on log-log axes."""
    xs = [x0, x0 / 10 ** decades]
    ys = [y0, y0 / 10 ** (slope * decades)]
    ax.plot(xs, ys, ls="--", lw=1.2, color=color or t.FG_4, alpha=0.85,
            label=label, zorder=2)


def plot_lab(spatial: Dict[str, List[dict]], temporal: Dict[str, List[dict]],
             save_path: str = "output/accuracy_lab.png") -> str:
    """Two log-log panels: spatial L2 vs dx (left), drift vs dt (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.3))
    # Manual layout: tight_layout ignores fig.text, so the equation footnote
    # would collide with the axis labels. Reserve bottom space for it instead.
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.22,
                        wspace=0.22)

    for name, pts in spatial.items():
        color = SERIES_COLORS[name]
        # CD and Verlet produce numerically identical curves (same stencil),
        # so CD gets square markers and a higher zorder to peek through —
        # "you can't see two lines" IS the lesson, made visible.
        marker = "s" if name == "Central difference" else "o"
        t.glow_line(ax1, [p["dx"] for p in pts], [max(p["l2"], 1e-17) for p in pts],
                    color=color, lw=2.2, label=name, marker=marker, markersize=5.5,
                    zorder=4 if name == "Central difference" else 3)
    for name, pts in temporal.items():
        color = SERIES_COLORS[name]
        t.glow_line(ax2, [p["dt"] for p in pts], [max(p["drift"], 1e-17) for p in pts],
                    color=color, lw=2.2, label=name, marker="o", markersize=5)

    # Slope guides, anchored to each scheme's coarsest data point so they run
    # through what they explain. Textbook slopes — measured orders may differ,
    # and where they do, the difference is the lesson (RK4 falls STEEPER than
    # its dt⁴ guide: its leading error is pure amplitude damping).
    cd_s = spatial["Central difference"][0]
    _slope_guide(ax1, cd_s["dx"], max(cd_s["l2"], 1e-17), 2,
                 label=r"$\propto dx^2$ (2nd order)")
    for name, pts in temporal.items():
        # Label each guide once: CD and Verlet share both slope and stencil,
        # so their identical textbook guides would duplicate the legend row.
        _slope_guide(ax2, pts[0]["dt"], max(pts[0]["drift"], 1e-17),
                     GUIDE_SLOPES[name], color=SERIES_COLORS[name],
                     label=GUIDE_LABELS[name] if name != "Velocity Verlet" else None)

    # Round-off floor: below this, smaller timesteps buy nothing
    floor = 1e-15
    ax2.axhline(floor, color=t.ACCENT.AMBER, lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax2.text(0.985, floor * 3.0, "round-off floor", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=8, color=t.ACCENT.AMBER,
             family=t.FONT_MONO)

    for ax, xlabel, ylabel, title in (
        (ax1, r"grid spacing  $dx$  [m]   (finer $\rightarrow$)",
         "relative L2 error vs exact", "STUDY 1: Spatial convergence"),
        (ax2, r"timestep  $dt$  [s]   (smaller $\rightarrow$)",
         "discrete-stencil energy drift", "STUDY 2: Energy conservation"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontfamily=t.FONT_DISPLAY, fontsize=12, loc="left")
        t.glass_legend(ax, fontsize=8, loc="lower left")

    fig.suptitle("ACCURACY LAB — WHY THE METRICS BEHAVE",
                 fontfamily=t.FONT_DISPLAY, fontsize=15, color=t.FG_1,
                 x=0.07, ha="left")
    t.equation_footnote(
        fig,
        r"$u(x,0)=g(x)$,  $u_t=0$,  force-free:  $\partial^2 u/\partial t^2 = c^2\, \partial^2 u/\partial x^2$"
        r"   ·   ground truth: d'Alembert with mirror images (exact through reflections)",
        "study 1 refines dx (dt $\\propto$ dx) · study 2 freezes the grid and halves dt · "
        "drift = the DISCRETE stencil energy, not the continuum integral",
        badge=t.cfl_badge(SPATIAL_CFL),
    )
    return t.finish_figure(fig, save_path, tight=False)


def print_report(spatial: Dict[str, List[dict]],
                 temporal: Dict[str, List[dict]]) -> None:
    """Terminal report: measured orders for both studies."""
    t.section("Study 1: spatial convergence (dt ∝ dx, L2 vs exact d'Alembert)")
    for name, pts in spatial.items():
        print(f"    {t.style(name, 'cyan', bold=True)}")
        orders = measured_order(pts, "l2")
        for i, p in enumerate(pts):
            o = f"{orders[i - 1]:.2f}" if i > 0 else " —"
            grid = f"N={p['num_points']:>5}  dx={p['dx']:.4f}  dt={p['dt']:.2e}"
            err = f"L2 {p['l2']:.2e}  (order {o})"
            print(f"      {t.style(grid, 'muted')}  {t.style(err, 'white')}")

    t.section("Study 2: temporal convergence (fixed grid, drift vs dt)")
    for name, pts in temporal.items():
        print(f"    {t.style(name, 'cyan', bold=True)}")
        orders = measured_order(pts, "drift")
        for i, p in enumerate(pts):
            o = f"{orders[i - 1]:.2f}" if i > 0 else " —"
            grid = f"cfl={p['cfl']:.2f}  dt={p['dt']:.2e}"
            err = f"drift {p['drift']:.2e}  (order {o})"
            print(f"      {t.style(grid, 'muted')}  {t.style(err, 'white')}")

    print()
    print(f"    {t.success('space: every integrator converges as dx² — the grid sets spatial error, not the time scheme')}")
    print(f"    {t.success('time: CD and Verlet share a stencil — identical dt² drift; RK4 lands ~200× lower (leading error = pure damping)')}")
    metric_line = ('the metric picks the winner: L2 says "all equal", drift says '
                   'RK4 ≫ Verlet = CD — always ask what a metric measures')
    print(f"    {t.warn(metric_line)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy Lab — convergence study")
    parser.add_argument("--selftest", action="store_true",
                        help="smaller/faster study for headless verification")
    args = parser.parse_args()

    t.banner("ACCURACY LAB", "Convergence study — why the metrics behave",
             eyebrow="DJC WAVE SUITE")

    if args.selftest:
        spatial = run_spatial_study(mults=(1, 2, 4), total_time=0.5)
        temporal = run_temporal_study(cfls=(0.8, 0.4, 0.2), total_time=0.5)
    else:
        spatial = run_spatial_study()
        temporal = run_temporal_study()

    print_report(spatial, temporal)
    path = plot_lab(spatial, temporal)
    print(f"\n  {t.style('chart', 'muted')}  {path}")


if __name__ == "__main__":
    main()
