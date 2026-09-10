"""
Crack the Whip — challenge mode.

The goal: drive the whip's tip past the speed of sound (Mach 1.0). You get a
parameter budget — taper ratio, taper exponent, and the energy of your throw
(pulse amplitude + velocity kick) — and every parameter costs energy budget.
Reach Mach 1 on budget for a bonus; the leaderboard remembers your best runs.

The physics scales like a real whip: tip speed grows with pulse energy and
with sqrt of the taper ratio, until numerical reality (and the fixed string
length) stops paying for more taper — so the skill is in balancing the knobs,
not maxing them.

Run:
    python whip_challenge.py              # interactive session
    python whip_challenge.py --demo       # run three preset attempts
    python whip_challenge.py --best       # show the leaderboard
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np

import djc_theme as t
from solver import VerletSolver
from string_model import String

__all__ = ["attempt_crack", "score_attempt", "run_attempt", "load_board", "save_attempt"]

BOARD_PATH = os.path.join("output", "whip_leaderboard.json")
SOUND_SPEED = 343.0
TOTAL_TIME = 2.0
POINTS = 1000

# ----------------------------------------------------------------------------
# Budget model — every knob spends from the same energy wallet
# ----------------------------------------------------------------------------

def energy_budget(params: dict) -> float:
    """Energy cost of a build (arbitrary units, normalized to ~1.0 = strong throw)."""
    return (
        0.30 * (params["taper_ratio"] / 1000.0)
        + 0.08 * max(params["taper_exponent"] - 1.0, 0.0)
        + 0.55 * (params["drive"] / 30.0) ** 2
        + 0.18 * (params["amplitude"] / 5.0) ** 2
    )


BUDGET_MAX = 1.30   # spending cap per attempt


def score_attempt(params: dict, mach: float) -> dict:
    """Score one attempt: Mach achievement + efficiency, Mach-1 bonus."""
    spent = energy_budget(params)
    score = 1000.0 * mach + 500.0 * max(0.0, 1.0 - spent / BUDGET_MAX)
    if mach >= 1.0:
        score += 500.0
    return {
        **{k: round(float(v), 3) for k, v in params.items()},
        "mach": round(float(mach), 3),
        "budget_spent": round(spent, 3),
        "score": round(score, 1),
        "cracked": mach >= 1.0,
    }


# ----------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------

def attempt_crack(
    taper_ratio: float,
    taper_exponent: float,
    drive: float,
    amplitude: float,
    width: float = 2.5,
    position: float = 15.0,
    num_points: int = 1000,
    total_time: float = TOTAL_TIME,
) -> dict:
    """
    Run one whip attempt and return tip-velocity statistics.

    The pulse at ``position`` carries a Gaussian displacement (``amplitude``)
    and a velocity kick (``drive`` m/s) — the hand's snap.
    """
    mu_base = 0.02
    w = String(
        length=100.0,
        num_points=num_points,
        tension=100.0,
        density_profile="tapered",
        density_base=mu_base,
        density_tip=mu_base / taper_ratio,
        taper_exponent=taper_exponent,
        boundary_right="free",          # the tip must be free to crack
    )
    w.set_initial_pulse(position=position, amplitude=amplitude, width=width,
                        velocity_kick=drive)

    solver = VerletSolver(w, enable_force=False)
    dt = 0.8 * w.dx / w.get_max_wave_speed()   # auto-CFL — can never abort
    solver.solve(total_time=total_time, dt=dt,
                 save_interval=max(1, int(0.0005 / dt)), verbose=False)

    tip_v = np.array([row[-1] for row in solver.velocity_history])
    times = np.array(solver.time_history[: len(tip_v)])
    peak = float(np.abs(tip_v).max())
    peak_i = int(np.argmax(np.abs(tip_v)))
    return {
        "mach": peak / SOUND_SPEED,
        "peak_tip_velocity": peak,
        "crack_time": float(times[peak_i]),
        "tip_velocity": tip_v,
        "time": times,
        "dt": dt,
    }


def run_attempt(params: dict, verbose: bool = True) -> dict:
    """Attempt + score in one call; returns the leaderboard entry."""
    result = attempt_crack(**params)
    entry = score_attempt(params, result["mach"])
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if verbose:
        _report(entry)
    return {**entry, "tip_velocity": result["tip_velocity"], "time": result["time"]}


def _report(entry: dict) -> None:
    mach = entry["mach"]
    spent = entry["budget_spent"]
    print()
    print(t.rule(64))
    print(f"  {t.style('THROW RESULT', 'cyan', bold=True)}")
    t.kv("Mach number", f"{mach:.3f}", color="amber" if mach >= 1.0 else "white")
    t.kv("Peak tip speed", f"{mach * SOUND_SPEED:.1f} m/s  (sound = {SOUND_SPEED:.0f} m/s)")
    over = spent > BUDGET_MAX
    t.kv("Budget spent", f"{spent:.2f} / {BUDGET_MAX:.2f}", color="red" if over else "white")
    print(f"    {t.meter(min(spent / BUDGET_MAX, 1.0), color='red' if over else None)}"
          + (t.style("  over budget!", "red", bold=True) if over else ""))
    t.kv("Score", f"{entry['score']:.0f}", color="green" if entry["cracked"] else "white")
    if entry["cracked"]:
        print(f"\n  {t.style('◆ CRACK! Sonic barrier broken.', 'green', bold=True)}")
    elif mach >= 0.7:
        print(f"\n  {t.style('▲ So close — the whip sang but did not crack.', 'amber')}")
    else:
        print(f"\n  {t.style('· The whip whispered. More energy, better taper.', 'muted')}")
    print(t.rule(64))


# ----------------------------------------------------------------------------
# Leaderboard — local JSON, newest first, ranked by score
# ----------------------------------------------------------------------------

def load_board() -> list:
    if os.path.exists(BOARD_PATH):
        try:
            with open(BOARD_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_attempt(entry: dict) -> list:
    board = load_board()
    slim = {k: v for k, v in entry.items() if k not in ("tip_velocity", "time")}
    board.append(slim)
    board.sort(key=lambda e: e.get("score", 0), reverse=True)
    board = board[:20]
    os.makedirs(os.path.dirname(BOARD_PATH), exist_ok=True)
    with open(BOARD_PATH, "w") as f:
        json.dump(board, f, indent=1)
    return board


def print_board(board: Optional[list] = None) -> None:
    board = board if board is not None else load_board()
    print()
    print(t.rule(64))
    print(f"  {t.style('◆ LEADERBOARD — top 20 by score', 'teal_bright', bold=True)}")
    print(t.rule(64))
    if not board:
        print(f"  {t.style('No attempts yet. Be the first to crack the whip.', 'muted')}")
        print(t.rule(64))
        return
    for i, e in enumerate(board, 1):
        mach = e.get("mach", 0.0)
        flag = t.style("CRACK", "green", bold=True) if e.get("cracked") else t.style("    ", "muted")
        rank = t.style(f"{i:2d}.", "muted")
        mach_s = t.style(f"Mach {mach:5.2f}", "amber" if mach >= 1 else "body")
        score_s = t.style(f"score {e.get('score', 0):6.0f}", "cyan")
        params_s = (f"R={e.get('taper_ratio', 0):4.0f} x={e.get('taper_exponent', 0):.1f} "
                    f"v={e.get('drive', 0):4.1f} A={e.get('amplitude', 0):.1f}")
        when = t.style(e.get("timestamp", ""), "rule")
        print(f"  {rank} {flag}  {mach_s}  {score_s}  {t.style(params_s, 'muted')}  {when}")
    print(t.rule(64))


# ----------------------------------------------------------------------------
# Interactive session
# ----------------------------------------------------------------------------

PROMPT_SPECS = [
    ("taper_ratio", "Taper ratio μ_base/μ_tip", 100.0, 5000.0, 1000.0),
    ("taper_exponent", "Taper exponent", 1.0, 3.0, 2.0),
    ("drive", "Velocity kick [m/s] (the snap)", 0.0, 40.0, 15.0),
    ("amplitude", "Pulse amplitude [m]", 0.5, 6.0, 2.5),
]


def interactive() -> None:
    print()
    print(t.rule(64))
    print(f"  {t.style('◆ CRACK THE WHIP', 'cyan', bold=True)}")
    print(f"  {t.style('Drive the whip tip past Mach 1.0 — on budget.', 'body', dim=True)}")
    print(t.rule(64))
    print(f"\n  {t.style('Physics intel:', 'teal_bright', bold=True)} tip speed scales with pulse "
          f"energy and with sqrt(taper ratio) — until extra taper stops paying.\n")

    while True:
        params = {}
        print(t.rule(56))
        for key, label, lo, hi, default in PROMPT_SPECS:
            raw = input(f"  {label} [{lo:g}–{hi:g}] (default {default:g}): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print(f"\n  {t.style('Later.', 'muted')}\n")
                return
            try:
                val = float(raw) if raw else default
            except ValueError:
                val = default
            params[key] = max(lo, min(hi, val))

        cost = energy_budget(params)
        if cost > BUDGET_MAX:
            print(t.error(f"over budget: {cost:.2f} / {BUDGET_MAX:.2f} — dial it back"))
            continue
        print(f"\n  {t.style(f'budget: {cost:.2f} / {BUDGET_MAX:.2f}', 'cyan')}")
        print(f"  {t.style('cracking…', 'muted')}")

        entry = run_attempt(params)
        save_attempt(entry)

        again = input(f"\n  {t.style('another throw? [Y/n]', 'body')} ").strip().lower()
        if again in ("n", "no"):
            break

    print_board()


def demo() -> None:
    """Three archetypal attempts: too weak, efficient, brute force."""
    print(f"\n  {t.style('DEMO — three archetypal throws', 'teal_bright', bold=True)}")
    presets = [
        ("The Gentle Flick", dict(taper_ratio=200, taper_exponent=1.5, drive=8.0, amplitude=2.0)),
        ("The Pro's Snap", dict(taper_ratio=1500, taper_exponent=2.0, drive=30.0, amplitude=5.5, width=1.8)),
        ("The Sledgehammer", dict(taper_ratio=3000, taper_exponent=3.0, drive=40.0, amplitude=6.0)),
    ]
    for name, params in presets:
        print(f"\n  {t.style('▸ ' + name, 'teal_bright', bold=True)}")
        entry = run_attempt(params)
        save_attempt(entry)
    print_board()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crack the Whip challenge")
    parser.add_argument("--demo", action="store_true", help="run three preset throws")
    parser.add_argument("--best", action="store_true", help="show the leaderboard")
    args = parser.parse_args()

    if args.best:
        print_board()
    elif args.demo:
        demo()
    else:
        interactive()
