"""
Sonification for the wave simulation.

Turns solver histories into sound using only the Python standard library
(``wave`` + ``math``): a probe node's velocity becomes a sine tone whose
frequency and amplitude follow the motion, so the string literally plays its
own dynamics. On whip runs this means you can *hear* the crack.

No third-party dependencies. Audio is written as 16-bit mono WAV and either
played through the default OS player or exported to a file.

Usage:
    from sonify import velocity_to_wav, sonify_run

    wav_path = velocity_to_wav(velocity_signal, dt, "output/whip.wav")
"""
import math
import struct
import subprocess
import sys
import wave
from typing import List, Optional, Sequence

import numpy as np

import djc_theme as t

__all__ = ["velocity_to_wav", "sonify_run", "play_wav", "whip_summary_sound"]


SAMPLE_RATE = 22050          # Hz — plenty for pitch mapping, small files
MAX_FREQ = 1200.0            # Hz — cap so fast physics still sounds musical
BASE_FREQ = 110.0            # Hz — A2 pedal tone for near-zero velocity
PLAY_SECONDS_CAP = 20.0      # never emit absurdly long audio


# ----------------------------------------------------------------------------
# Signal shaping
# ----------------------------------------------------------------------------

def _normalize(signal: np.ndarray) -> np.ndarray:
    """Map any signal to [-1, 1] by its peak (silence-safe)."""
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak < 1e-12:
        return np.zeros_like(signal)
    return signal / peak


def velocity_to_wav(
    velocity: Sequence[float],
    dt: float,
    path: str,
    *,
    base_freq: float = BASE_FREQ,
    max_freq: float = MAX_FREQ,
    soft_clip: bool = True,
) -> str:
    """
    Render a velocity history as a WAV file.

    |v| maps to pitch (base → max) and to loudness with a soft envelope, so a
    sharp velocity spike (a whip crack, a string snap) reads as a percussive
    event rather than a smeared tone. Sine + soft saturation keeps it pleasant.

    Returns the path written.
    """
    v = np.asarray(velocity, dtype=float)
    if len(v) < 2 or dt <= 0:
        raise ValueError("need a velocity signal with at least 2 samples and dt > 0")

    duration = min(len(v) * dt, PLAY_SECONDS_CAP)
    if duration < 0.05:
        duration = 0.05
    n_samples = int(duration * SAMPLE_RATE)

    # Stretch the physics signal onto the audio timeline.
    src_t = np.linspace(0.0, len(v) * dt, len(v))
    out_t = np.linspace(0.0, duration, n_samples, endpoint=False)
    v_audio = np.interp(out_t, src_t, v)

    env = _normalize(np.abs(v_audio))

    # Pitch: frequency follows the *slower* envelope so the tone stays musical
    # while loudness carries the sharp transients.
    kernel = max(1, int(0.004 * SAMPLE_RATE))  # ~4 ms smoothing
    smooth_env = np.convolve(env, np.ones(kernel) / kernel, mode="same")
    freqs = base_freq + (max_freq - base_freq) * smooth_env

    phase = 2.0 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    tone = np.sin(phase)

    # Soft saturation for a bit of "crack" without harsh clipping.
    if soft_clip:
        tone = np.tanh(2.0 * tone)

    samples = np.int16(np.clip(tone * env, -1.0, 1.0) * 32767 * 0.85)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return path


# ----------------------------------------------------------------------------
# Playback (OS default player; export always works, playing is best-effort)
# ----------------------------------------------------------------------------

_PLAYERS = ["aplay", "paplay", "afplay", "ffplay", "cvlc"]


def play_wav(path: str) -> bool:
    """Try to play a WAV through the first available OS player. Best-effort."""
    import shutil

    for player in _PLAYERS:
        if shutil.which(player):
            try:
                flags = ["-q", "-nodisp", "-autoexit"] if player == "ffplay" else []
                subprocess.run([player, *flags, path], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                return True
            except Exception:
                continue
    print(t.warn(f"no audio player found (tried: {', '.join(_PLAYERS)}) — file saved instead"))
    return False


def sonify_run(
    time_history: Sequence[float],
    velocity_history: Sequence[np.ndarray],
    node_index: int,
    out_path: str,
    *,
    label: str = "sonification",
    play: bool = True,
) -> Optional[str]:
    """
    Sonify one solver run at ``node_index`` and export/play the result.

    Returns the WAV path, or None when the history is unusable.
    """
    if not velocity_history:
        print(t.warn(f"{label}: nothing to sonify (empty velocity history)"))
        return None

    v = np.array([row[node_index] for row in velocity_history], dtype=float)
    times = np.asarray(time_history[: len(v)], dtype=float)
    if len(v) < 2:
        print(t.warn(f"{label}: too few samples to sonify"))
        return None
    dt = float(times[1] - times[0]) if len(times) > 1 else 1.0

    velocity_to_wav(v, dt, out_path)
    peak = float(np.max(np.abs(v)))
    print(t.success(f"{label} → {out_path}  (peak |v| = {peak:.2f}, "
                    f"{min(len(v) * dt, PLAY_SECONDS_CAP):.1f}s audio)"))
    if play:
        play_wav(out_path)
    return out_path


def whip_summary_sound(stats: dict, out_path: str) -> str:
    """A tiny synthesized 'crack' flourish whose loudness follows the Mach number."""
    mach = float(stats.get("mach_number", 0.0))
    dur = 0.6
    n = int(dur * SAMPLE_RATE)
    ts = np.arange(n) / SAMPLE_RATE

    # Two quick descending chirps: the snap and its echo.
    f0 = 900.0 + 600.0 * min(mach, 2.0) / 2.0
    chirp = np.sin(2 * np.pi * (f0 * ts - 3000.0 * ts**2))
    echo = 0.4 * np.sin(2 * np.pi * (0.6 * f0 * ts - 1200.0 * ts**2))
    env = np.exp(-ts * 9.0)
    samples = np.int16(np.clip((chirp + echo) * env * min(0.2 + 0.4 * mach, 0.9), -1, 1) * 32767 * 0.8)

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    print(t.success(f"crack flourish → {out_path}  (Mach {mach:.2f})"))
    return out_path


if __name__ == "__main__":
    # CLI: sonify the whip run's tip, or a plain string probe.
    import os

    import solver as solver_mod
    from string_model import String

    os.makedirs("output", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "whip"
    play = "--no-play" not in sys.argv

    if which == "whip":
        whip = String(length=100.0, num_points=1000, tension=100.0, density_profile="tapered",
                      density_base=0.02, density_tip=0.002, taper_exponent=1.5,
                      boundary_right="free")
        whip.set_initial_pulse(position=10.0, amplitude=2.0, width=3.0)
        s = solver_mod.VerletSolver(whip, enable_force=False)
        s.solve(total_time=2.0, dt=0.0004, save_interval=25, verbose=False)
        sonify_run(s.time_history, s.velocity_history, node_index=-1,
                   out_path="output/whip_crack.wav", label="whip tip", play=play)
    else:
        s0 = String(length=50.0, num_points=400, tension=60.0, density_profile="uniform",
                    density_uniform=0.01)
        s0.set_initial_gaussian(center=25.0, width=4.0, amplitude=1.0)
        s = solver_mod.VerletSolver(s0, enable_force=False)
        s.solve(total_time=3.0, dt=0.001, save_interval=4, verbose=False)
        sonify_run(s.time_history, s.velocity_history, node_index=200,
                   out_path="output/string_tone.wav", label="string probe", play=play)
