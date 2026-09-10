"""
Tests for the Break It On Purpose missions.

Covers the pure scoring functions on synthetic trial lists, progress-file
persistence, and one *fast* physics trial per mission (coarse grids, short
times) so the suite stays quick. The precise physics thresholds (CFL razor
edge, wall-breach amplitude, Mach crossover) are pinned by the mission demo
in missions.py --demo, not here — these tests pin the shape of the scoring
and the trial plumbing.
"""
import json
import os

import pytest

import missions


# ----------------------------------------------------------------------------
# Progress file
# ----------------------------------------------------------------------------

def test_progress_roundtrip(tmp_path):
    path = str(tmp_path / "progress.json")
    missions.save_mission_result("m1", score=820.0, won=True, attempts=3, path=path)
    prog = missions.load_progress(path)
    assert prog["m1"]["best"] == 820.0
    assert prog["m1"]["won"] is True
    assert prog["m1"]["attempts"] == 3

    # Best score never regresses; attempts accumulate.
    missions.save_mission_result("m1", score=500.0, won=False, attempts=2, path=path)
    prog = missions.load_progress(path)
    assert prog["m1"]["best"] == 820.0
    assert prog["m1"]["won"] is True
    assert prog["m1"]["attempts"] == 5


def test_progress_missing_file_is_empty(tmp_path):
    assert missions.load_progress(str(tmp_path / "nope.json")) == {}


# ----------------------------------------------------------------------------
# Scoring: M1 DETONATOR
# ----------------------------------------------------------------------------

def test_m1_no_blowup_scores_zero():
    trials = [{"blew_up": False, "cfl": 0.8, "max_abs": 1.0}]
    assert missions.score_m1(trials) == 0.0


def test_m1_shallow_overshoot_outranks_brute_force():
    shallow = [{"blew_up": True, "cfl": 1.02, "max_abs": 1e3},
               {"blew_up": False, "cfl": 0.95, "max_abs": 1.0}]
    brute = [{"blew_up": True, "cfl": 2.0, "max_abs": 1e30}]
    assert missions.score_m1(shallow) > missions.score_m1(brute)
    # 600 + 300 (CFL 1.02) + 100 edge bonus = 994 — the only 1000 would be
    # detonation *at* CFL 1.00, which survives exactly: unreachable by design.
    assert missions.score_m1(shallow) == pytest.approx(994.0)


def test_m1_max_achievable_score_is_900():
    # cfl 0.1 → full overshoot bonus, no edge ride: 600 + 300 = 900. The
    # remaining 100 requires surviving a ride at CFL >= 0.9 in the same session.
    assert missions.score_m1([{"blew_up": True, "cfl": 0.1, "max_abs": 1e9}]) \
        == pytest.approx(900.0)
    assert missions.score_m1([{"blew_up": True, "cfl": 0.1, "max_abs": 1e9},
                              {"blew_up": False, "cfl": 1.0, "max_abs": 1.0}]) \
        == pytest.approx(missions.MAX_SCORE)


# ----------------------------------------------------------------------------
# Scoring: M2 WALL BREAKER
# ----------------------------------------------------------------------------

def test_m2_no_breach_scores_zero():
    trials = [{"breached": False, "min_interior": 0.2, "amp": 4.0}]
    assert missions.score_m2(trials) == 0.0


def test_m2_precision_beats_depth():
    # Shallow breach at minimal amplitude vs deep breach at brute-force amp.
    deep = [{"breached": True, "min_interior": -1.0, "amp": 12.0}]
    precise = [{"breached": True, "min_interior": -0.2, "amp": 8.5}]
    assert missions.score_m2(precise) > missions.score_m2(deep)


def test_m2_cap():
    # Minimal shove + deep breach saturates all three components.
    trials = [{"breached": True, "min_interior": -5.0, "amp": 0.5}]
    assert missions.score_m2(trials) == pytest.approx(missions.MAX_SCORE)


# ----------------------------------------------------------------------------
# Scoring: M3 SONIC TAPER
# ----------------------------------------------------------------------------

def test_m3_no_crack_scores_zero():
    trials = [{"cracked": False, "mach": 0.935, "exponent": 2.0, "ratio": 2000.0}]
    assert missions.score_m3(trials) == 0.0


def test_m3_first_trial_crack_scores_more_than_lucky_fourth():
    one_shot = [{"cracked": True, "mach": 1.10}]
    grind = [{"cracked": False, "mach": 0.5}] * 3 + [{"cracked": True, "mach": 1.10}]
    assert missions.score_m3(one_shot) > missions.score_m3(grind)


def test_m3_higher_mach_scores_higher():
    # mach 1.02 first-try already saturates 700+300+20 → cap; use two grindy
    # cracks below the cap to show the mach bonus ranks scores monotonically.
    low = [{"cracked": False, "mach": 0.5}] * 2 + [{"cracked": True, "mach": 1.02}]
    high = [{"cracked": False, "mach": 0.5}] * 2 + [{"cracked": True, "mach": 1.60}]
    assert missions.score_m3(high) > missions.score_m3(low)
    # ... and a first-try crack is capped.
    assert missions.score_m3([{"cracked": True, "mach": 1.60}]) \
        == pytest.approx(missions.MAX_SCORE)


# ----------------------------------------------------------------------------
# Fast physics: one real trial per mission
# ----------------------------------------------------------------------------

def test_m1_trial_marginal_survives_and_over_mult_explodes():
    survived = missions.run_m1_trial(1.0, num_points=101, total_time=0.3)
    assert not survived["blew_up"]
    assert survived["cfl"] == pytest.approx(0.8, abs=0.01)

    # mult 3.0 → CFL 2.4: twice the propagation speed the grid can carry.
    # Blowup needs time to grow from roundoff, hence the longer run.
    blown = missions.run_m1_trial(3.0, num_points=101, total_time=1.0)
    assert blown["blew_up"]
    assert blown["cfl"] == pytest.approx(2.4, abs=0.03)


def test_m2_trial_amp4_contained_amp12_breaches():
    contained = missions.run_m2_trial(4.0, num_points=101, total_time=0.5)
    assert not contained["detonated"]
    assert not contained["breached"]

    breached = missions.run_m2_trial(12.0, num_points=101, total_time=0.5)
    assert not breached["detonated"]
    assert breached["breached"]
    assert breached["min_interior"] < 0.0


def test_m3_trial_gentle_exponent_cracks_steep_fails():
    # Fixed marginal throw: exponent 1.0 cracks, exponent 2.0 falls short.
    gentle = missions.run_m3_trial(1.0, ratio=2000.0)
    steep = missions.run_m3_trial(2.0, ratio=2000.0)
    assert gentle["cracked"] and gentle["mach"] > 1.0
    assert not steep["cracked"] and steep["mach"] < 1.0
    assert gentle["mach"] > steep["mach"]


# ----------------------------------------------------------------------------
# Mission registry integrity
# ----------------------------------------------------------------------------

def test_all_missions_are_well_formed():
    for mid in ("m1", "m2", "m3"):
        spec = missions.MISSIONS[mid]
        assert spec["name"] and spec["brief"] and spec["hint"] and spec["takeaway"]
        assert callable(spec["run"]) and callable(spec["score"]) and callable(spec["win"])
        # A no-win trial list must not crash win/score.
        assert spec["win"]([]) is False
        assert spec["score"]([]) == 0.0
