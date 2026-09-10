"""
Break It On Purpose — three guided destruction missions for the wave simulator.

Each mission hands you a system that is *guaranteed* to survive, and a knob
guaranteed to destroy it. Your job: find the edge, cross it deliberately, and
bank the physics lesson. Every mission ends with a one-line takeaway and a
score (max 1000, capped — precision pays more than brute force):

    M1  DETONATOR      Double the timestep until the grid explodes.
                       CFL 1.00 survives forever; the cliff is just above it.
    M2  WALL BREAKER   Crank the pulse amplitude until it punches through the
                       Lennard-Jones repulsive wall into forbidden territory.
    M3  SONIC TAPER    A marginal whip throw, one knob that matters: shape the
                       taper until the tip breaks Mach 1.

Progress (best score per mission) is kept in output/missions_progress.json.

Run:
    python missions.py                     # mission menu
    python missions.py m1 mult=2.0         # one-shot trial (key=value overrides)
    python missions.py --demo              # scripted expert arcs for all three
    python missions.py --board             # progress summary
    python missions.py --reset             # wipe progress
"""
import argparse
import json
import os
import warnings as _warnings
from datetime import datetime
from typing import Optional

import numpy as np

import djc_theme as t
from solver import VerletSolver
from string_model import String
from whip_challenge import attempt_crack

__all__ = [
    "run_m1_trial", "run_m2_trial", "run_m3_trial",
    "score_m1", "score_m2", "score_m3",
    "load_progress", "save_mission_result",
]

PROGRESS_PATH = os.path.join("output", "missions_progress.json")
SOUND_SPEED = 343.0

# Calibrated Lennard-Jones well — same derivation as main.py (u* = 2 m well,
# softness omega0 = 5 rad/s). Kept local so importing missions.py never runs
# main.py's module side effects.
U_STAR, OMEGA0 = 2.0, 5.0
K2_CAL = OMEGA0**2 * U_STAR**8 / 36.0
K1_CAL = K2_CAL * U_STAR**6 / 2.0

MAX_SCORE = 1000.0


# ----------------------------------------------------------------------------
# Progress file
# ----------------------------------------------------------------------------

def load_progress(path: str = PROGRESS_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_mission_result(mission_id: str, score: float, won: bool,
                        attempts: int, path: str = PROGRESS_PATH) -> dict:
    """Merge one finished session into the progress file and return it."""
    prog = load_progress(path)
    prev = prog.get(mission_id, {})
    prog[mission_id] = {
        "best": max(float(prev.get("best", 0.0)), round(float(score), 1)),
        "won": bool(prev.get("won", False)) or bool(won),
        "attempts": int(prev.get("attempts", 0)) + int(attempts),
        "last": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prog, f, indent=1)
    return prog


# ----------------------------------------------------------------------------
# M1 — DETONATOR: violate the CFL condition on purpose
# ----------------------------------------------------------------------------

def run_m1_trial(mult: float, *, length: float = 50.0, num_points: int = 500,
                 total_time: float = 2.0) -> dict:
    """
    Integrate a uniform string (no force term) with dt = mult x (safe dt),
    where the safe step is 0.8 dx/c — so CFL = 0.8 x mult.

    A trial "blows up" when the solution goes NaN or exceeds 50x the initial
    amplitude (grid-scale bloom, long before the overflow).
    """
    s = String(length=length, num_points=num_points, tension=50.0,
               density_profile="uniform", density_uniform=0.01)
    s.set_initial_gaussian(center=length / 2.0, width=5.0, amplitude=1.0)

    dt = mult * 0.8 * s.dx / s.get_max_wave_speed()
    cfl = s.get_max_wave_speed() * dt / s.dx

    solver = VerletSolver(s, enable_force=False)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        hist = solver.solve(total_time=total_time, dt=dt, save_interval=20,
                            verbose=False, check_cfl=False)
    u = np.asarray(hist, dtype=float)

    if np.isnan(u).any():
        max_abs = float("inf")
        blew = True
    else:
        max_abs = float(np.max(np.abs(u)))
        blew = max_abs > 50.0
    return {"mission": "m1", "mult": float(mult), "cfl": float(cfl),
            "dt": float(dt), "blew_up": blew, "max_abs": max_abs}


def score_m1(trials: list) -> float:
    """Detonate for 600, up to +300 for the shallowest overshoot (CFL 2 -> 1),
    +100 for having first survived a ride on the edge (CFL >= 0.9)."""
    blow = [x for x in trials if x["blew_up"]]
    if not blow:
        return 0.0
    score = 600.0 + 300.0 * max(0.0, min((2.0 - min(x["cfl"] for x in blow)) / 1.0, 1.0))
    if any((not x["blew_up"]) and x["cfl"] >= 0.9 for x in trials):
        score += 100.0
    return min(score, MAX_SCORE)


# ----------------------------------------------------------------------------
# M2 — WALL BREAKER: punch through the Lennard-Jones repulsive wall
# ----------------------------------------------------------------------------

def run_m2_trial(amp: float, *, length: float = 50.0, num_points: int = 500,
                 total_time: float = 2.5) -> dict:
    """
    Launch a Gaussian pulse (width 2.0 m) from rest on a string whose
    displacement sits at the well equilibrium u* = 2 m, with the calibrated
    Lennard-Jones force active. "Breach" = any interior node dips below
    u = 0, i.e. the pulse climbed over the 1/u^12 repulsive wall.
    """
    sigma = 2.0
    center = length / 2.0

    def ic(x):
        return U_STAR + amp * np.exp(-((x - center) ** 2) / (2 * sigma**2))

    s = String(length=length, num_points=num_points, tension=50.0,
               density_profile="uniform", density_uniform=0.01)
    s.set_initial_custom(displacement_func=ic)

    solver = VerletSolver(s, enable_force=True, k1=K1_CAL, k2=K2_CAL)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        hist = solver.solve(total_time=total_time, dt=0.0005, save_interval=20,
                            verbose=False, check_cfl=False)
    u = np.asarray(hist, dtype=float)

    if np.isnan(u).any():
        return {"mission": "m2", "amp": float(amp), "detonated": True,
                "breached": False, "min_interior": float("nan")}
    # Exclude the boundary-stiffness artifact zone at each end.
    edge = max(5, num_points // 25)
    interior = u[:, edge:-edge]
    min_int = float(np.min(interior))
    return {"mission": "m2", "amp": float(amp), "detonated": False,
            "breached": min_int < 0.0, "min_interior": min_int}


def score_m2(trials: list) -> float:
    """Breach for 500, up to +300 for precision (breaching at the lowest
    amplitude — the minimal shove that crosses the wall is the skill),
    up to +200 by depth (|min u| past the wall)."""
    breach = [x for x in trials if x.get("breached")]
    if not breach:
        return 0.0
    best = 0.0
    for x in breach:
        depth = abs(x["min_interior"])
        precision = max(0.0, min((11.0 - x["amp"]) / 3.0, 1.0))
        s = 500.0 + 300.0 * precision + 200.0 * min(depth, 1.0)
        best = max(best, s)
    return min(best, MAX_SCORE)


# ----------------------------------------------------------------------------
# M3 — SONIC TAPER: shape the taper until the tip breaks Mach 1
# ----------------------------------------------------------------------------

def run_m3_trial(exponent: float, ratio: float = 2000.0) -> dict:
    """
    A marginal throw is fixed (amplitude 2.5 m, snap 8 m/s, narrow pulse,
    taper ratio 2000 unless overridden). You shape the taper; the physics
    decides whether the tip cracks Mach 1.
    """
    r = attempt_crack(taper_ratio=ratio, taper_exponent=exponent,
                      drive=8.0, amplitude=2.5, width=1.2)
    mach = float(r["mach"])
    return {"mission": "m3", "exponent": float(exponent), "ratio": float(ratio),
            "mach": mach, "cracked": mach >= 1.0}


def score_m3(trials: list) -> float:
    """Crack for 700, +25 per trial saved below four, +1000 x (Mach - 1)."""
    cracks = [x for x in trials if x.get("cracked")]
    if not cracks:
        return 0.0
    first = trials.index(cracks[0]) + 1          # trials used to first crack
    mach = max(x["mach"] for x in cracks)
    score = 700.0 + 100.0 * max(0, 4 - first) + 1000.0 * max(0.0, mach - 1.0)
    return min(score, MAX_SCORE)


# ----------------------------------------------------------------------------
# Mission metadata
# ----------------------------------------------------------------------------

MISSIONS = {
    "m1": {
        "name": "DETONATOR",
        "brief": ("The solver's safe timestep is 0.8 dx/c (CFL 0.8). Multiply it — "
                  "double dt, quadruple, whatever — until the grid explodes."),
        "knobs": [("mult", "dt multiplier vs safe step", 0.25, 4.0, 2.0)],
        "win": lambda trials: any(x["blew_up"] for x in trials),
        "score": score_m1,
        "run": lambda params: run_m1_trial(params["mult"]),
        "hint": "CFL 1.00 (mult 1.25) survives forever — the cliff is just above it.",
        "takeaway": ("CFL <= 1 is a wall, not a suggestion: at c*dt/dx = 1.00 the wave "
                     "crosses exactly one cell per step and information stays put; one "
                     "percent past it, error doubles every step until the grid detonates."),
    },
    "m2": {
        "name": "WALL BREAKER",
        "brief": ("The string rests at the well equilibrium u* = 2 m. Gentle pulses "
                  "rebound off the Lennard-Jones wall — crank the amplitude until the "
                  "pulse punches through into u < 0."),
        "knobs": [("amp", "Pulse amplitude [m]", 0.5, 12.0, 4.0)],
        "win": lambda trials: any(x["breached"] for x in trials),
        "score": score_m2,
        "run": lambda params: run_m2_trial(params["amp"]),
        "hint": "Contained pulses bounce off the well floor near u ~ 0.04. It takes a focused hit to climb the wall — try amp 8-12.",
        "takeaway": ("The 1/u^12 repulsive wall is stiff, not absolute: enough pulse energy "
                     "climbs it into forbidden territory — and the solver's force clamp, "
                     "not physics, is what finally stops you."),
    },
    "m3": {
        "name": "SONIC TAPER",
        "brief": ("A marginal throw is locked in (amp 2.5 m, snap 8 m/s). Shape the "
                  "taper — exponent and ratio — until the tip breaks Mach 1. "
                  "Budget your attempts: fewer tries, higher score."),
        "knobs": [("exponent", "Taper exponent", 1.0, 3.0, 2.0),
                  ("ratio", "Taper ratio", 100.0, 4000.0, 2000.0)],
        "win": lambda trials: any(x["cracked"] for x in trials),
        "score": score_m3,
        "run": lambda params: run_m3_trial(params["exponent"], params["ratio"]),
        "hint": "Ratio barely matters on this throw — try the exponent, and go gentler, not steeper.",
        "takeaway": ("Taper ratio is the marketing, taper shape is the physics: a gentle, "
                     "linear taper (exponent 1) keeps the pulse coherent all the way to "
                     "the tip — every steepening sheds it into reflections en route."),
    },
}

_ALIASES = {"detonator": "m1", "wall": "m2", "wallbreaker": "m2",
            "sonic": "m3", "sonictaper": "m3", "taper": "m3"}


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------

def _verdict_line(trial: dict) -> str:
    m = trial["mission"]
    if m == "m1":
        if trial["blew_up"]:
            peak = trial["max_abs"]
            peak_s = "inf" if np.isinf(peak) else f"{peak:.2e}"
            return t.error(f"DETONATED — solution peaked at {peak_s} (initial peak was 1.0)")
        return t.success(f"survived — bounded (max |u| = {trial['max_abs']:.4f}), "
                         f"CFL = {trial['cfl']:.2f}")
    if m == "m2":
        if trial["detonated"]:
            return t.error("DETONATED — overdriven; the well fight itself went unstable")
        if trial["breached"]:
            return t.success(f"WALL BREACHED — pulse punched through to "
                             f"u = {trial['min_interior']:+.3f} m")
        return t.warn(f"contained — rebounded off the well floor "
                      f"(min u = {trial['min_interior']:+.3f} m)")
    # m3
    mach = trial["mach"]
    if trial["cracked"]:
        return t.success(f"CRACK! tip reached Mach {mach:.3f}")
    return t.warn(f"no crack — peak Mach {mach:.3f} (need 1.000)")


def _finish_mission(mid: str, trials: list, started: int) -> None:
    spec = MISSIONS[mid]
    won = spec["win"](trials)
    score = spec["score"](trials)
    print()
    print(t.rule(64))
    if won:
        print(f"  {t.style('◆ MISSION COMPLETE', 'green', bold=True)}")
        t.kv("Score", f"{score:.0f} / {MAX_SCORE:.0f}", color="green")
        print()
        print(f"  {t.style('TAKEAWAY', 'teal_bright', bold=True)}")
        print(f"  {t.style(spec['takeaway'], 'body')}")
        save_mission_result(mid, score, True, len(trials))
    else:
        print(f"  {t.style('▲ MISSION INCOMPLETE', 'amber', bold=True)}")
        print(f"  {t.style(spec['hint'], 'body')}")
        save_mission_result(mid, 0.0, False, len(trials))
    print(t.rule(64))


# ----------------------------------------------------------------------------
# Interactive session
# ----------------------------------------------------------------------------

def play_mission(mid: str) -> None:
    spec = MISSIONS[mid]
    prog = load_progress().get(mid, {})
    best = prog.get("best", 0.0)

    print()
    print(t.rule(64))
    print(f"  {t.style('◆ M2 ' if False else '◆ MISSION ' + spec['name'], 'cyan', bold=True)}")
    print(f"  {t.style(spec['brief'], 'body', dim=True)}")
    if best:
        print(f"  {t.style(f'personal best: {best:.0f}', 'muted')}")
    print(t.rule(64))

    trials = []
    while True:
        print()
        params = {}
        for key, label, lo, hi, default in spec["knobs"]:
            raw = input(f"  {label} [{lo:g}-{hi:g}] (default {default:g}, q = back): ").strip()
            if raw.lower() in ("q", "quit", "back"):
                return
            try:
                val = float(raw) if raw else default
            except ValueError:
                val = default
            params[key] = max(lo, min(hi, val))

        print(f"  {t.style('running...', 'muted')}")
        trial = spec["run"](params)
        trials.append(trial)
        print(f"  {_verdict_line(trial)}")

        if spec["win"](trials):
            _finish_mission(mid, trials, len(trials))
            return


# ----------------------------------------------------------------------------
# One-shot CLI:  python missions.py m1 mult=2.0
# ----------------------------------------------------------------------------

def one_shot(mid: str, overrides: dict) -> None:
    spec = MISSIONS[mid]
    params = {key: default for key, _label, _lo, _hi, default in spec["knobs"]}
    for key, val in overrides.items():
        if key not in params:
            print(t.error(f"unknown knob '{key}' for {mid} "
                          f"(have: {', '.join(params)})"))
            raise SystemExit(2)
        params[key] = val
    trial = spec["run"](params)
    print(f"  {_verdict_line(trial)}")
    _finish_mission(mid, [trial], 1)


# ----------------------------------------------------------------------------
# Demo — scripted expert arcs
# ----------------------------------------------------------------------------

def demo() -> None:
    print(f"\n  {t.style('DEMO — three expert destruction arcs', 'teal_bright', bold=True)}")

    print(f"\n  {t.style('▸ M1 DETONATOR: ride the edge, then double dt', 'teal_bright', bold=True)}")
    trials = [run_m1_trial(1.25), run_m1_trial(2.0)]   # CFL 1.00, then CFL 1.6
    for x in trials:
        print(f"  {_verdict_line(x)}")
    _finish_mission("m1", trials, len(trials))

    print(f"\n  {t.style('▸ M2 WALL BREAKER: contained, then through', 'teal_bright', bold=True)}")
    trials = [run_m2_trial(4.0), run_m2_trial(10.0)]
    for x in trials:
        print(f"  {_verdict_line(x)}")
    _finish_mission("m2", trials, len(trials))

    print(f"\n  {t.style('▸ M3 SONIC TAPER: steep fails, gentle cracks', 'teal_bright', bold=True)}")
    print(f"  {t.style('running attempt 1 (exponent 2.0)...', 'muted')}")
    trials = [run_m3_trial(2.0)]
    print(f"  {_verdict_line(trials[0])}")
    print(f"  {t.style('running attempt 2 (exponent 1.0)...', 'muted')}")
    trials.append(run_m3_trial(1.0))
    print(f"  {_verdict_line(trials[1])}")
    _finish_mission("m3", trials, len(trials))


def board() -> None:
    prog = load_progress()
    print()
    print(t.rule(64))
    print(f"  {t.style('◆ BREAK IT ON PURPOSE — progress', 'teal_bright', bold=True)}")
    print(t.rule(64))
    if not prog:
        print(f"  {t.style('No missions played yet. Start with the DETONATOR.', 'muted')}")
        print(t.rule(64))
        return
    for mid in ("m1", "m2", "m3"):
        spec = MISSIONS[mid]
        e = prog.get(mid, {})
        won = e.get("won", False)
        flag = t.style("CLEARED", "green", bold=True) if won else t.style("UNPLAYED" if not e else "attempts only", "muted")
        score = e.get("best", 0.0)
        t.kv(f"{mid} {spec['name']}", f"{flag}   best {score:.0f}",
             color="green" if won else "muted")
        if won:
            print(f"    {t.style('lesson learned: ' + spec['takeaway'][:72] + '...', 'muted', dim=True)}")
    print(t.rule(64))


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Break It On Purpose missions")
    parser.add_argument("mission", nargs="?", help="m1 | m2 | m3 (or a name alias)")
    parser.add_argument("overrides", nargs="*", help="key=value knob overrides for one-shot")
    parser.add_argument("--demo", action="store_true", help="scripted expert arcs")
    parser.add_argument("--board", action="store_true", help="show progress")
    parser.add_argument("--reset", action="store_true", help="wipe progress file")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
            print(t.success("progress wiped"))
        return
    if args.board:
        board()
        return
    if args.demo:
        demo()
        return

    if args.mission:
        mid = _ALIASES.get(args.mission.lower(), args.mission.lower())
        if mid not in MISSIONS:
            parser.error(f"unknown mission '{args.mission}' (choose m1, m2, m3)")
        overrides = {}
        for item in args.overrides:
            if "=" not in item:
                parser.error(f"override '{item}' must be key=value")
            k, v = item.split("=", 1)
            overrides[k] = float(v)
        one_shot(mid, overrides)
        return

    # Full menu
    print()
    print(t.rule(64))
    print(f"  {t.style('◆ BREAK IT ON PURPOSE', 'cyan', bold=True)}")
    print(f"  {t.style('Three systems that survive everything — until you decide otherwise.', 'body', dim=True)}")
    print(t.rule(64))
    for mid in ("m1", "m2", "m3"):
        spec = MISSIONS[mid]
        e = load_progress().get(mid, {})
        cleared = t.style("cleared", "green") if e.get("won") else t.style("open", "muted")
        print(f"  {t.style(mid, 'cyan', bold=True)}  {t.style(spec['name'], 'white', bold=True)}"
              f"   [{cleared}]")
    print()
    choice = input("  mission (m1/m2/m3, q to quit): ").strip().lower()
    if choice in ("q", "quit", ""):
        print(f"\n  {t.style('Come back reckless.', 'muted')}\n")
        return
    mid = _ALIASES.get(choice, choice)
    if mid not in MISSIONS:
        print(t.error(f"unknown mission '{choice}'"))
        return
    play_mission(mid)


if __name__ == "__main__":
    main()
