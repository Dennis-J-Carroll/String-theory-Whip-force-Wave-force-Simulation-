"""
The Playground — a live, playable wave simulator.

One themed matplotlib window: the string animates continuously while sliders
(tension, well equilibrium, well stiffness, damping, pulse amplitude) re-solve
the system in real time on a small fast grid. Energy bars pulse with the
exchange between kinetic and elastic energy — split into elastic strain vs
well energy (button or 'w') so you can feel how wave speed changes what the
pulse carries: crank tension and the elastic share grows with c² while the
well share ignores it. A probe sparkline traces a single node's motion, and a
live equation strip shows the actual numbers in play — including a CFL badge
that tells you how close you are to the stability edge.

Hotkeys:  space = pause/resume · r = reset pulse · s = export sonification ·
w = toggle elastic/well split · c = ride the CFL edge (dial → 1.00) ·
x = cross the edge (dial → 1.60) · q = quit

The Courant dial is the torture instrument: the sim auto-times itself at CFL
0.8, but the dial lets you push dt past the stability limit on purpose. Below
1.0 the string behaves; past 1.0 grid-scale noise grows every step until the
blow-up guard freezes the sim with an explanation. Dial back or press Reset
to recover — that one minute of play teaches stability better than any
footnote.

Run:  python playground.py            (window)
      python playground.py --selftest (headless render check → output/playground_preview.png)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.widgets import Button, Slider

import djc_theme as t
from string_model import String
from solver import VerletSolver

t.apply()  # idempotent

# ----------------------------------------------------------------------------
# Simulation kernel — small grid for real-time interaction
# ----------------------------------------------------------------------------
GRID_POINTS = 200          # spatial resolution (fast grid)
STRING_LENGTH = 50.0       # m
BUFFER_STEPS = 3000        # rolling history length for the animation
CFL_TARGET = 0.8           # dt = CFL_TARGET * dx / max(c) — never unstable
PROBE_INDEX = GRID_POINTS // 2

# Wave-force well calibration (same scheme as main.py)
U_STAR_DEFAULT = 2.0       # m — well equilibrium
OMEGA0_DEFAULT = 5.0       # rad/s — well stiffness


def calibrate(u_star: float, omega0: float):
    """LJ constants from the well's equilibrium depth-scale and curvature."""
    k2 = omega0**2 * u_star**8 / 36.0
    k1 = k2 * u_star**6 / 2.0
    return k1, k2


def well_energy(u, density, dx, k1: float, k2: float, u_star: float) -> float:
    """Energy stored against the Lennard-Jones well, in joules.

    W(u) = k1/u^12 − k2/u^6 is the per-unit-mass potential whose gradient is
    the solver's force term; the rest state u* contributes zero by subtracting
    W(u*), so ∫μ·W dx is the conserved ledger's well term measured on absolute
    displacement (no |u − u*| reflection — that is a different potential).
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        uu = np.maximum(np.asarray(u, dtype=float), 1e-6)
        v = k1 / uu**12 - k2 / uu**6
        v_rest = k1 / u_star**12 - k2 / u_star**6
        integrand = np.asarray(density, dtype=float) * (v - v_rest)
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=1e9, neginf=0.0)
    return float(np.trapz(integrand, dx=dx))


def make_world(tension: float, u_star: float, omega0: float,
               damping: float, pulse_amp: float,
               courant: float = CFL_TARGET):
    """Build a fresh string + solver + timestep from slider values.

    dt = courant * dx / c_max. The dial can push courant past 1.0 — that is
    the torture mode; the blow-up guard in the tick loop is the safety net.
    """
    k1, k2 = calibrate(u_star, omega0)
    s = String(
        length=STRING_LENGTH,
        num_points=GRID_POINTS,
        tension=tension,
        density_profile="uniform",
        density_uniform=0.01,
    )
    s.set_initial_custom(
        displacement_func=lambda x: u_star + pulse_amp * np.exp(-((x - STRING_LENGTH / 2) ** 2) / (2 * 5.0**2)),
    )
    solver = VerletSolver(s, enable_force=True, k1=k1, k2=k2, damping=damping)
    dt = courant * s.dx / s.get_max_wave_speed()
    return s, solver, dt


# ----------------------------------------------------------------------------
# The Playground window
# ----------------------------------------------------------------------------

def run_playground(selftest: bool = False) -> None:
    fig = plt.figure(figsize=(13, 8), facecolor=t.BG_BASE)
    fig.canvas.manager.set_window_title("DJC Wave Playground") if hasattr(fig.canvas, "manager") and fig.canvas.manager else None

    # --- Layout -------------------------------------------------------------
    # Main wave axes (left, tall), right column: energy bars + probe sparkline,
    # a live equation strip below, sliders at the bottom.
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        left=0.07, right=0.97, top=0.90, bottom=0.41,
        width_ratios=[3.2, 1.0], height_ratios=[2.6, 1.0, 0.7],
        hspace=0.45, wspace=0.18,
    )
    ax_wave = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_probe = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")

    x = np.linspace(0.0, STRING_LENGTH, GRID_POINTS)

    # --- State --------------------------------------------------------------
    state = {
        "paused": False,
        "detonated": False,
        "e0": None,
        "u_hist": [],
        "ke_hist": [],
        "pe_hist": [],
        "v_probe_hist": [],
        "steps_done": 0,
        "u": None,
        "v": None,
    }

    def rebuild(*_args, **_):
        """Rebuild the world from slider values (called on any slider change).

        Accepts (and ignores) the positional value Slider.on_changed passes."""
        s, solver, dt = make_world(
            tension=sliders["Tension"].val,
            u_star=sliders["Well equilibrium"].val,
            omega0=sliders["Well stiffness"].val,
            damping=sliders["Damping"].val,
            pulse_amp=sliders["Pulse amplitude"].val,
            courant=sliders["Courant dial"].val,
        )
        state["string"], state["solver"], state["dt"] = s, solver, dt
        state["detonated"] = False
        state["e0"] = s.get_total_energy()
        state["u_hist"], state["ke_hist"], state["pe_hist"] = [], [], []
        state["w_hist"] = []
        state["v_probe_hist"], state["steps_done"] = [], 0
        state["u"], state["v"] = s.displacement.copy(), s.velocity.copy()
        k1, k2 = calibrate(sliders["Well equilibrium"].val, sliders["Well stiffness"].val)
        state["k1"], state["k2"] = k1, k2
        well_line.set_ydata(np.full_like(x, sliders["Well equilibrium"].val))

    # --- Artists ------------------------------------------------------------
    (main_line,) = ax_wave.plot([], [], color=t.ACCENT.CYAN, lw=2.2, zorder=4)
    (trail_line,) = ax_wave.plot([], [], color=t.ACCENT.CYAN, lw=6, alpha=0.25, zorder=3)
    (well_line,) = ax_wave.plot(x, np.full_like(x, U_STAR_DEFAULT), color=t.ACCENT.VIOLET,
                                lw=1.2, ls="--", alpha=0.55, zorder=2, label="well floor u*")
    ax_wave.set_xlim(0, STRING_LENGTH)
    ax_wave.set_ylim(0.0, 6.0)
    ax_wave.set_xlabel("Position  x  [m]")
    ax_wave.set_ylabel("Displacement  u(x, t)")
    ax_wave.set_title("DJC WAVE PLAYGROUND", fontfamily=t.FONT_DISPLAY, fontsize=14, loc="left")
    t.glass_legend(ax_wave, loc="upper right", fontsize=8)

    # Energy bars: kinetic / elastic strain / well. The elastic-vs-well
    # split (button or 'w') is the lesson: elastic grows with tension (c²),
    # the well term does not care about T at all.
    state["split"] = True
    bar_labels = ["Kinetic", "Elastic", "Well"]
    bars = ax_energy.barh([2, 1, 0], [0.5, 0.5, 0.5], height=0.55,
                          color=[t.ACCENT.VIOLET, t.TEAL.N300, t.ACCENT.AMBER], alpha=0.9)
    ax_energy.set_yticks([2, 1, 0], bar_labels, fontsize=9)
    ax_energy.set_xlim(0, 1)
    ax_energy.set_xticks([])
    ax_energy.set_title("Energy exchange", fontfamily=t.FONT_DISPLAY, fontsize=11)

    (probe_line,) = ax_probe.plot([], [], color=t.ACCENT.CYAN, lw=1.5)
    ax_probe.set_xlim(0, BUFFER_STEPS)
    ax_probe.set_ylim(-0.5, 0.5)
    ax_probe.set_xlabel(f"history (node {PROBE_INDEX})", fontsize=8)
    ax_probe.set_ylabel("probe v", fontsize=8)
    ax_probe.tick_params(labelsize=7)

    info_text = ax_info.text(
        0.02, 0.98, "", va="top", ha="left", fontsize=9,
        family=t.FONT_MONO, color=t.FG_2, transform=ax_info.transAxes,
    )

    # --- Sliders ------------------------------------------------------------
    slider_specs = [
        ("Tension", 10.0, 200.0, 60.0),
        ("Well equilibrium", 0.5, 5.0, U_STAR_DEFAULT),
        ("Well stiffness", 1.0, 15.0, OMEGA0_DEFAULT),
        # Slight default dissipation bleeds grid-scale modal fuzz; set to 0
        # for a perfectly conservative system.
        ("Damping", 0.0, 0.5, 0.05),
        ("Pulse amplitude", 0.0, 3.0, 0.5),
        # The torture dial: CFL number = c*dt/dx. Past 1.0 the scheme is
        # unstable — noise grows every step until the blow-up guard intervenes.
        ("Courant dial", 0.05, 2.0, CFL_TARGET),
    ]
    sliders = {}
    slider_ax_height = 0.030
    gap = 0.011
    y = 0.335
    for name, lo, hi, init in slider_specs:
        sax = fig.add_axes([0.14, y, 0.62, slider_ax_height])
        sl = Slider(sax, name, lo, hi, valinit=init, valstep=(hi - lo) / 200.0)
        # Theme the slider: navy track, teal knob, Space Grotesk label.
        sl.label.set_color(t.FG_2)
        sl.label.set_fontsize(9)
        sl.ax.set_facecolor(t.NAVY.N800)
        for spine in sl.ax.spines.values():
            spine.set_color("#223246")
        if hasattr(sl, "track"):
            sl.track.set_color(t.NAVY.N700)
        if hasattr(sl, "handle"):
            sl.handle.set_facecolor(t.TEAL.N500)
            sl.handle.set_edgecolor(t.ACCENT.CYAN)
        sl.valtext.set_color(t.ACCENT.CYAN)
        sl.valtext.set_fontsize(8)
        sl.on_changed(rebuild)
        sliders[name] = sl
        y -= (slider_ax_height + gap)

    # --- Buttons: pause / reset --------------------------------------------
    bax_pause = fig.add_axes([0.80, 0.335, 0.075, 0.036])
    bax_reset = fig.add_axes([0.885, 0.335, 0.075, 0.036])
    btn_pause = Button(bax_pause, "Pause", hovercolor="0.85")
    btn_reset = Button(bax_reset, "Reset", hovercolor="0.85")
    for btn, ax_b in ((btn_pause, bax_pause), (btn_reset, bax_reset)):
        btn.label.set_color(t.FG_1)
        btn.label.set_fontsize(9)
        ax_b.set_facecolor(t.NAVY.N700)
        for spine in ax_b.spines.values():
            spine.set_color("#223246")

    def toggle_pause(_):
        state["paused"] = not state["paused"]
        btn_pause.label.set_text("Play" if state["paused"] else "Pause")

    def reset(_):
        rebuild()
        btn_pause.label.set_text("Pause")

    btn_pause.on_clicked(toggle_pause)
    btn_reset.on_clicked(reset)

    # Split toggle — three bars (kinetic / elastic / well) or the classic
    # merged two-bar view. Default ON: the split is the teaching view.
    bax_split = fig.add_axes([0.80, 0.290, 0.16, 0.036])
    btn_split = Button(bax_split, "Elastic | Well", hovercolor="0.85")
    btn_split.label.set_color(t.FG_1)
    btn_split.label.set_fontsize(9)
    bax_split.set_facecolor(t.NAVY.N700)
    for spine in bax_split.spines.values():
        spine.set_color("#223246")

    def toggle_split(_):
        state["split"] = not state["split"]
        btn_split.label.set_text("KE + PE merged" if state["split"] else "Elastic | Well")
        fig.canvas.draw_idle()

    btn_split.on_clicked(toggle_split)

    # --- Live equation strip (bottom band) ----------------------------------
    # The governing PDE with the actual current constants substituted in.
    # Mathtext is re-rendered only when a slider moves, never per frame.
    ax_eq = fig.add_axes([0.07, 0.005, 0.90, 0.075])
    ax_eq.axis("off")
    eq_text = ax_eq.text(
        0.0, 1.0, "", va="top", ha="left", fontsize=11, color=t.FG_2,
        family=t.FONT_SANS,
    )

    def update_equation_strip():
        k1, k2 = state["k1"], state["k2"]
        s_ = state["string"]
        c_max = float(np.max(s_.wave_speed))
        eq_text.set_text(
            r"$\frac{\partial^2 u}{\partial t^2} = c^2\,\frac{\partial^2 u}{\partial x^2} "
            r"+ \frac{12\,k_1}{u^{13}} - \frac{6\,k_2}{u^{7}}$"
            f"        c = √(T/$\\mu$) = {c_max:6.1f} m/s"
            f"     k₁ = {k1:.2e}    k₂ = {k2:.2e}"
            f"     $\\gamma$ = {sliders['Damping'].val:.2f}"
            "\n"
            r"$\mathrm{wave\ equation}\;+\;$Lennard-Jones well (equilibrium "
            f"u* = {sliders['Well equilibrium'].val:.2f} m, stiffness $\\omega_0$ = {sliders['Well stiffness'].val:.1f} rad/s)"
            + (
                "\n"
                "split view — the elastic bar scales with T (it rides c²): raise Tension and it outruns the amber well bar,\n"
                "whose energy depends only on k₁, k₂, u*. Toggle: button or 'w'."
                if state["split"] else ""
            )
        )
        fig.canvas.draw_idle()

    _orig_rebuild = rebuild

    def rebuild(*args, **kwargs):
        _orig_rebuild(*args, **kwargs)
        update_equation_strip()

    # --- Keyboard shortcuts -------------------------------------------------
    def on_key(event):
        if event.key == " ":
            toggle_pause(None)
        elif event.key == "r":
            reset(None)
        elif event.key == "q":
            plt.close(fig)
        elif event.key == "s":
            export_sonification()
        elif event.key == "w":
            toggle_split(None)
        elif event.key == "c":
            # Ride the stability edge: CFL = 1.0 is marginal for the wave
            # equation — noise neither grows nor decays.
            sliders["Courant dial"].set_val(1.0)
        elif event.key == "x":
            # Cross the edge on purpose: watch the detonation, then recover.
            sliders["Courant dial"].set_val(1.6)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def export_sonification():
        from sonify import velocity_to_wav
        path = "output/playground_probe.wav"
        os.makedirs("output", exist_ok=True)
        if len(state["v_probe_hist"]) > 2:
            velocity_to_wav(np.array(state["v_probe_hist"]), state["dt"], path)
            info_text.set_text(info_text.get_text() + f"\naudio → {path}")
            fig.canvas.draw_idle()

    # --- Animation tick -----------------------------------------------------
    SUBSTEPS = 12  # physics steps per frame — smooth motion at small dt

    def tick(_frame):
        if not state.get("paused", False) and not state.get("detonated", False):
            solver = state["solver"]
            s = state["string"]
            u, v = state["u"], state["v"]
            for _ in range(SUBSTEPS):
                u, v = solver.step(u, v, state["dt"], s.dx, s.wave_speed)
                s.displacement, s.velocity = u, v
                s.apply_boundary_conditions()
                u, v = s.displacement.copy(), s.velocity.copy()
                state["steps_done"] += 1
                # Blow-up guard: CFL > 1 grows grid-scale noise exponentially;
                # freeze with an explanation instead of rendering NaN soup.
                if (not np.all(np.isfinite(u))) or np.max(np.abs(u)) > 100.0:
                    state["detonated"] = True
                    break
                # Record every few substeps to keep the buffers long-lived
                if state["steps_done"] % 4 == 0:
                    ke = s.get_kinetic_energy()
                    pe = s.get_potential_energy()
                    we = well_energy(u, s.density, s.dx, state["k1"], state["k2"],
                                     sliders["Well equilibrium"].val)
                    state["u_hist"].append(u.copy())
                    state["ke_hist"].append(ke)
                    state["pe_hist"].append(pe)
                    state["w_hist"].append(we)
                    state["v_probe_hist"].append(v[PROBE_INDEX])
                    if len(state["u_hist"]) > BUFFER_STEPS:
                        for key in ("u_hist", "ke_hist", "pe_hist", "w_hist", "v_probe_hist"):
                            state[key].pop(0)
            state["u"], state["v"] = u, v

        # --- redraw ----------------------------------------------------------
        if state["u_hist"]:
            u_now = state["u_hist"][-1]
            main_line.set_data(x, u_now)
            trail_line.set_data(x, u_now)
            u_floor = sliders["Well equilibrium"].val
            ax_wave.set_ylim(min(u_now.min(), u_floor) - 0.4,
                             max(u_now.max(), u_floor) + 0.8)

            ke, pe_el = state["ke_hist"][-1], state["pe_hist"][-1]
            we = state["w_hist"][-1] if state["w_hist"] else 0.0
            if state["split"]:
                scale = max(ke + pe_el + we, 1e-9)
                bars[0].set_width(min(ke / scale, 1.0))
                bars[1].set_width(min(pe_el / scale, 1.0))
                # Escape dips put the well ledger briefly negative (energy
                # handed back); clamp so the bar reads zero, not negative.
                bars[2].set_width(min(max(we, 0.0) / scale, 1.0))
            else:
                scale = max(ke + pe_el, 1e-9)
                bars[0].set_width(min(ke / scale, 1.0))
                bars[1].set_width(min(pe_el / scale, 1.0))
                bars[2].set_width(0.0)

            vp = state["v_probe_hist"]
            probe_line.set_data(np.arange(len(vp)), vp)
            if vp:
                # Scale to the recent window so decayed motion stays readable
                recent = vp[-500:]
                lo, hi = min(recent), max(recent)
                pad = max(0.05, 0.15 * (hi - lo))
                ax_probe.set_ylim(lo - pad, hi + pad)

            s = state["string"]
            courant = sliders["Courant dial"].val
            e_now = s.get_total_energy()
            d_e = (e_now - state["e0"]) / max(abs(state["e0"]), 1e-300)
            if state.get("detonated", False):
                info_text.set_color(t.ACCENT.RED)
                info_text.set_text(
                    "DETONATED\n"
                    f"Courant = {courant:.2f} > 1.0\n"
                    "grid noise grows each step.\n"
                    "Dial back below 1.0\n"
                    "or press Reset to recover."
                )
            else:
                info_text.set_color(t.FG_2)
                d_e_flag = "" if abs(d_e) < 0.05 else "  !"
                info_text.set_text(
                    f"c  = {np.max(s.wave_speed):7.1f} m/s\n"
                    f"dt = {state['dt']*1000:6.2f} ms\n"
                    f"Courant {courant:5.2f} {'OK' if courant <= 1.0 else 'UNSTABLE'}\n"
                    f"dE = {d_e*100:+7.2f}%{d_e_flag}\n"
                    f"well share = {100*we/max(ke + pe_el + we, 1e-9):5.1f}%\n"
                    f"u* = {sliders['Well equilibrium'].val:4.2f} m\n"
                    f"steps = {state['steps_done']}"
                )
        return main_line, trail_line, probe_line, bars[0], bars[1], info_text

    rebuild()  # initialize world from slider defaults

    if selftest:
        # Headless verification, in two acts:
        # Act 1 — stable regime: advance and render the dressed frame.
        for _ in range(600):
            tick(None)
        assert np.all(np.isfinite(state["u"])), "stable run produced non-finite state"
        assert abs(state["u"].max()) < 100.0, "stable run tripped the blow-up guard"
        fig.savefig("output/playground_preview.png", dpi=120, facecolor=t.BG_BASE)
        print(t.success("playground selftest → output/playground_preview.png"))

        # Act 2 — torture: push past CFL 1.0, expect detonation, expect recovery.
        sliders["Courant dial"].set_val(1.6)
        detonated = False
        for _ in range(400):
            tick(None)
            if state.get("detonated", False):
                detonated = True
                break
        assert detonated, "blow-up guard never fired at CFL 1.6"
        print(t.success("torture dial: detonation guard fired at CFL 1.6"))
        sliders["Courant dial"].set_val(0.8)  # rebuild clears the flag
        for _ in range(50):
            tick(None)
        assert not state.get("detonated", False) and np.all(np.isfinite(state["u"])), \
            "sim did not recover after dialing Courant back"
        print(t.success("recovery confirmed after dialing back to CFL 0.8"))

        # Act 3 — the energy-split lesson: elastic strain scales with tension
        # (it rides c²), while the well term depends only on k₁, k₂, u*.
        # This is why the toggle exists: wave speed changes what a pulse
        # carries, and the bars must prove it.
        def _ledger(tension_value):
            s, _solver, _dt = make_world(tension=tension_value, u_star=2.0,
                                         omega0=5.0, damping=0.0, pulse_amp=0.5)
            pe = s.get_potential_energy()
            we = well_energy(s.displacement, s.density, s.dx,
                             *calibrate(2.0, 5.0), u_star=2.0)
            return pe, we

        pe_lo, we_lo = _ledger(60.0)
        pe_hi, we_hi = _ledger(180.0)
        assert we_hi == pytest.approx(we_lo, abs=1e-9), \
            "well energy must not depend on tension"
        assert pe_hi / pe_lo == pytest.approx(180.0 / 60.0, rel=1e-6), \
            "elastic energy must scale exactly with T"
        assert state["split"] and bars[2].get_width() >= 0.0
        print(t.success("energy split verified: elastic ∝ T, well tension-free"))

        plt.close(fig)
        return

    # Animated mode — keep a reference so FuncAnimation isn't garbage collected.
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, tick, interval=30, blit=False, cache_frame_data=False)

    print(t.style("Playground live — space pauses, r resets, s saves audio, q quits.",
                  "teal_bright", bold=True))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DJC Wave Playground")
    parser.add_argument("--selftest", action="store_true",
                        help="headless render check, no window")
    args = parser.parse_args()
    run_playground(selftest=args.selftest)
