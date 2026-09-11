# Wave Dynamics Simulation: Classical String Mechanics & Wave Forces

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Physics](https://img.shields.io/badge/physics-inspired-purple.svg)]()
[![Status](https://img.shields.io/badge/status-exploratory%20research-orange.svg)]()

![Wave Simulation](docs/wave_animation.gif)
*Note: Run simulation to generate visualization*

## 🌊 Why This Exists

> **"The laws of physics provide principled frameworks for understanding computation."**

This isn't just a physics simulation—it's a **thinking laboratory** for developing computational intuition. By exploring wave propagation, energy distribution, and resonance patterns in physical systems, I build intuition for:

- **Information flow** in neural networks (wave propagation ↔ gradient flow)
- **Attention mechanisms** in transformers (energy distribution ↔ attention weights)
- **Training stability** in deep learning (energy conservation ↔ loss stability)
- **Interpretability** through physics-inspired design (observables ↔ mechanistic understanding)

**The pattern:** *The why? -----> apply*<br>
Understand physics deeply → Extract mathematical principles → Apply to ML architecture design → Build more interpretable, robust AI systems.

**See [ML_CONNECTIONS.md](ML_CONNECTIONS.md) for explicit connections between this simulation and modern machine learning architectures.**

---

## Overview

This project implements a comprehensive numerical simulation of **classical wave mechanics** on strings with variable properties. Despite the repository name referencing "String Theory," this is a **classical physics simulation** focusing on:

- **Wave propagation** on strings with spatially-varying properties
- **Whip mechanics** - modeling the famous "whip crack" supersonic tip phenomenon
- **Modified wave equations** with Lennard-Jones-like potentials for wave interactions
- **Energy conservation** in non-linear wave systems

### ⚡ Important Clarification: Classical vs. Quantum

**This is NOT quantum string theory** (the fundamental physics theory proposing 1-dimensional strings as basic constituents of matter). This simulation models **classical continuous strings** (like ropes, cables, or whips) governed by Newtonian mechanics and the classical wave equation.

**This is exploratory research** - programming physics ideas with the intent to imaginatively experiment and build computational intuition. 

## Features

### Physics Engine
- ✅ Multiple numerical integrators: Central Difference, RK4, Velocity Verlet
- ✅ Vectorized NumPy implementation for performance
- ✅ Lennard-Jones-like potential for modeling wave interactions
- ✅ Linear density tapering for whip simulations
- ✅ CFL stability condition checking
- ✅ Real-time energy conservation monitoring

### Visualization & Analysis — DJC Design System
- **Themed plots**: deep-navy canvas with teal/cyan glow lines, glass legends, and Orbitron headings — the same aesthetic as [dennisjcarroll.com](https://dennisjcarroll.com)
- **Type stack matches the site**: Orbitron (display) · Space Grotesk (body) · Fira Code (mono) — loaded from Google Fonts in the dashboard and HTML report, bundled as matplotlib fonts for the static plots
- **Equation footnotes**: every chart carries its governing equation (`∂²u/∂t² = c²∂²u/∂x² + F(u)`, `c = √(T/μ)`, the LJ force) with the actual constants substituted in, plus a green/red **CFL badge** (`c·dt/dx` vs the stability limit) — see `djc_theme.equation_footnote` / `cfl_badge`
- **Signature colormaps**: `djc_wave` (teal → electric cyan), `djc_diverging` (violet / navy / cyan) and `djc_time` (slate past → glowing present) for wave evolution overlays
- Wave evolution plots with customizable time steps
- Energy conservation plots (Kinetic, Potential, Total)
- Phase space trajectory visualization
- Space-time heatmaps for wave history
- Animated GIF generation
- Potential and force function plots
- Themed console output (ANSI teal/cyan banners, drift meters)

All theming lives in `djc_theme.py` — the single source of truth for palette, fonts and console styling. Import it and call `apply()` to theme any new plot. Brand fonts are fetched once from Google Fonts and cached under `~/.cache/djc_theme/fonts`; set `DJC_THEME_NO_FONTS=1` to stay fully offline (falls back to DejaVu).

### Interactive (Plotly) Surfaces — same look, live charts
- **`djc_plotly.py`** bridges the design system to Plotly: a registered `djc_dark` template (navy canvas, teal/cyan colorway, glass legends), the signature colorscales (`WAVE_SCALE`, `DIVERGING_SCALE`, `TIME_SCALE`) and `glow_traces` — the glow line, as Plotly traces
- **`interactive_visualization.py`**: animated wave (themed Play/Pause + time slider), 3D space-time surface, potential & force, energy monitor, phase space — all sharing the DJC look and series colors (KE violet · PE teal · Total cyan, matching the static charts)
- **`dashboard_app.py`**: a real-time Dash dashboard (`python dashboard_app.py` → http://localhost:8050) with the calibrated Lennard-Jones well as defaults (u* = 2 m, ω₀ = 5 rad/s — the cosmological 10³⁵ defaults detonated), a **CFL badge** wired to a dt-as-fraction-of-limit slider (past 1.0 it refuses to run, playground rules), k₂ offsets that move the well without moving the equilibrium, and honest energies (KE + elastic + well on the absolute displacement — the conserved ledger, drift ≈ 0.06%)
- **`interactive_demo.py`**: generates all six themed Plotly artifacts as HTML under `output/` (first one offline-capable with plotly.js embedded)
- **`wave_solver(...)`** in `solver.py`: the legacy positional API `dashboard_app`/`interactive_demo` were written against (it did not exist), implemented on the modern String/Verlet physics — free ends by default so the string can rest at u*

### Software Engineering
- Object-oriented design with String and Solver classes
- YAML configuration files for parameter management
- Comprehensive unit tests with pytest
- Clean modular structure

## The Mathematics

This simulation solves the **1D wave equation with external forces**:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} + F(u)$$

Where:
- $u(x,t)$ is the transverse displacement
- $c = \sqrt{T/\mu(x)}$ is the wave speed (tension $T$ / linear density $\mu$)
- $F(u)$ is the external force derived from a potential

### For Whip Simulations

The "whip crack" occurs due to **linear density tapering**. As the wave travels toward the tip where $\mu(x) \to 0$, the wave speed increases dramatically:

$$c(x) = \sqrt{\frac{T}{\mu(x)}} \implies c(x) \to \infty \text{ as } \mu(x) \to 0$$

This causes the wave velocity to exceed the speed of sound, creating the characteristic "crack."

### The Lennard-Jones-Like Potential

The force term is derived from a potential function:

$$V(u) = k_1 \left(\frac{1}{u^{12}}\right) - k_2 \left(\frac{1}{u^{6}}\right)$$

$$F(u) = -\frac{dV}{du} = 12k_1 \left(\frac{1}{u^{13}}\right) - 6k_2 \left(\frac{1}{u^{7}}\right)$$

- **Repulsive term** ($k_1/u^{12}$): Prevents wave collapse at small displacements
- **Attractive term** ($k_2/u^{6}$): Models binding forces at larger scales

## Project Structure

```
.
├── config.yaml          # Simulation configuration parameters
├── constants.py         # Physical constants and parameters (deprecated, use config.yaml)
├── djc_theme.py         # DJC Design System: palette, fonts, colormaps, console styling
├── djc_plotly.py        # DJC Design System bridge for Plotly (template + scales)
├── string_model.py      # String class for wave properties
├── solver.py            # Numerical solvers (Central Diff, RK4, Verlet) + wave_solver shim
├── visualization.py     # Plotting and visualization functions (themed)
├── analysis.py          # Energy tracking and phase space analysis (themed)
├── interactive_visualization.py  # Plotly interactive charts (themed)
├── dashboard_app.py     # Real-time Dash dashboard (themed, calibrated physics)
├── interactive_demo.py  # Generates the six themed Plotly HTML artifacts
├── main.py              # Main simulation script
├── playground.py        # Live slider-driven playground (Courant torture dial included)
├── sonify.py            # Stdlib-only sonification: hear the wave / the crack
├── report.py            # Self-contained interactive HTML report generator
├── whip_challenge.py    # Gamified Mach-1 challenge with local leaderboard
├── missions.py          # Break It On Purpose: guided destruction missions
├── accuracy_lab.py      # Convergence study: why the metrics behave
├── tests/               # Unit tests
│   ├── test_solver.py
│   ├── test_string.py
│   ├── test_energy.py
│   └── test_accuracy_lab.py
└── docs/                # Generated animations and plots
```

## Installation

1. Clone the repository:
```bash
git clone git@github.com:Dennis-J-Carroll/String-theory-Whip-force-Wave-force-Simulation-.git
cd String-theory-Whip-force-Wave-force-Simulation-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

Run the default simulation:
```bash
python main.py
```

### The Playground (interactive)

A live, playable window — drag sliders and watch the physics respond in real time:
```bash
python playground.py
```
- Sliders: tension, LJ well equilibrium u\*, well stiffness ω₀, damping, pulse amplitude
- Live energy bars, probe-node sparkline, and an equation strip showing the actual constants in play (with a CFL stability badge)
- **Elastic-vs-well split** (default on, toggle with the button or `w`): three bars — kinetic (violet), elastic strain (teal), well energy (amber) — plus a live "well share" readout. The lesson: elastic energy rides c², so raising tension grows the teal bar while the amber well bar doesn't move; crank tension and feel how wave speed changes what the pulse carries
- Hotkeys: `space` pause · `r` reset · `s` export the probe signal as audio · `w` toggle elastic/well split · `q` quit
- `python playground.py --selftest` renders a headless preview to `output/playground_preview.png`

### Sonification (hear the physics)

Map solver velocity histories to audio with zero third-party dependencies:
```bash
python sonify.py whip     # the tip of the whip — you can hear the crack
python sonify.py string   # a probe node on the uniform string
python sonify.py whip --no-play   # just export output/whip_crack.wav
```

### Interactive HTML report (share the simulation)

`main.py` writes `output/wave_report.html` on every run — a single self-contained
file (data as JSON, `<canvas>` renderer, zero dependencies, works from `file://`).
Recipients can scrub time, play/pause, hover the wave for exact `(x, u, t)`
readouts, and click/drag the space-time heatmap to extract the wavefront shape
at any instant (shown as an amber ghost curve).

Runs driven by the calibrated LJ well also embed an **escape analysis** — the
same well-depth vs crest-energy readout as the dashboard's PHYSICS card: the
exact escape threshold (−V(u\*)), the pulse crest's energy split (well +
elastic), an escape ratio with a 100 px color-coded bar (green bound / amber
≥ 85 % near escape / red ≥ 100 % unbound), and a dotted amber `V = 0` threshold
line on the wave chart. Force-free runs omit the panel entirely.

Generate a standalone demo:
```bash
python report.py --out output/wave_report.html
```

### Crack the Whip (challenge mode)

Drive the whip tip past **Mach 1.0** on a parameter budget — taper ratio,
taper exponent, pulse amplitude and velocity kick (the hand's snap) all draw
from the same energy wallet. Overspend and you lose the efficiency bonus;
the leaderboard remembers your best throws:
```bash
python whip_challenge.py --demo   # three archetypal throws
python whip_challenge.py          # interactive session
python whip_challenge.py --best   # leaderboard (output/whip_leaderboard.json)
```
The physics scales like a real whip: tip speed grows with pulse energy and
~√(taper ratio) — until extra taper stops paying and skill beats brute force.

### The Accuracy Lab (why the metrics behave)

A two-part convergence study that turns "drift < 5%" from an arbitrary
pass/fail into understanding. It re-solves one initial condition against the
**exact d'Alembert solution** (method of images, exact through reflections):

- **Study 1 — spatial**: refine the grid with dt ∝ dx and every integrator
  lands on ~dx² — the grid, not the time scheme, sets spatial error.
- **Study 2 — temporal**: freeze the grid and halve dt. Now they split:
  central difference and Verlet share a stencil and drift identically at ~dt²,
  while RK4 lands ~200× lower (its leading error on linear waves is pure
  amplitude damping, ~dt⁵).

The deepest lesson: **the metric picks the winner** — L2 says "all equal",
drift says RK4 ≫ Verlet = CD. (The lab's first run exposed a real solver bug —
a first-order leapfrog bootstrap silently capping central difference at 1st
order — now fixed and regression-tested.)
```bash
python accuracy_lab.py            # full study -> output/accuracy_lab.png
python accuracy_lab.py --selftest # smaller/faster study
```

### The Courant torture dial (feel the stability limit)

The playground's auto-timestep makes instability impossible by design — so
the new **Courant dial** (0.05–2.0) lets you break it on purpose. Push past
1.0 and watch grid-scale noise bloom every step until the blow-up guard
freezes the sim with a red explanation; dial back or Reset to recover.

- `c` — ride the edge (Courant → 1.00)
- `x` — cross the edge (Courant → 1.60), then recover
- `dE` in the info panel shows energy drift live, `!` flags > 5%

### Break It On Purpose (guided missions)

Three guided destruction missions — each hands you a system that survives
everything and one knob guaranteed to destroy it. Find the edge, cross it
deliberately, bank the lesson. Progress (best score per mission) is kept in
`output/missions_progress.json`.

```bash
python missions.py --demo            # scripted expert arcs for all three
python missions.py                   # mission menu (interactive)
python missions.py m1 mult=2.0       # one-shot trial, key=value overrides
python missions.py --board           # progress summary
python missions.py --reset           # wipe progress
```

- **M1 DETONATOR** — double the timestep until the grid explodes. Ride the
  edge first (CFL ≥ 0.9) for bonus points; shallower overshoot scores higher
  than brute force.
- **M2 WALL BREAKER** — crank the pulse amplitude until it punches through
  the Lennard-Jones wall into u < 0. The minimal shove that crosses scores
  more than the biggest hammer.
- **M3 SONIC TAPER** — a marginal throw is locked in; shape the taper until
  the tip breaks Mach 1. The counter-intuitive lesson: the taper *exponent*
  is the real lever, not the ratio.

### Custom Configuration

Edit `config.yaml` to modify simulation parameters:
```yaml
simulation:
  integrator: "rk4"  # Options: "central_diff", "rk4", "verlet"

string:
  length: 50.0
  num_points: 500
  tension: 50.0
  density_type: "uniform"  # Options: "uniform", "tapered"
```

Then run:
```bash
python main.py --config config.yaml
```

### The Dash Dashboard (real-time web UI)

```bash
python dashboard_app.py        # → http://localhost:8050
```

Live sliders for the well stiffness (k₂ offsets that keep the equilibrium at
u* = 2 m), wave speed, CFL fraction, grid and pulse shape — with a green/red
CFL badge, a themed animated wave, 3D surface, energy monitor, phase space and
the potential/force curves. Requires the Plotly extras:
`pip install dash dash-bootstrap-components plotly` (also in requirements.txt).

To regenerate the static interactive artifacts instead:

```bash
python interactive_demo.py     # → output/*.html (animation, 3D, energies, ...)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Examples

### Example 1: Classical Wave on Uniform String
```python
from string_model import String
from solver import CentralDifferenceSolver
from visualization import animate_wave

# Create uniform string
string = String(length=50, num_points=500, tension=50, density_profile='uniform')

# Initialize with Gaussian pulse
string.set_initial_gaussian(center=25, width=5, amplitude=1.0)

# Solve
solver = CentralDifferenceSolver(string)
solution = solver.solve(total_time=5.0, dt=0.01)

# Visualize
animate_wave(solution, save_path='docs/wave.gif')
```

### Example 2: Whip Crack Simulation
```python
# Create tapered string (whip)
whip = String(length=100, num_points=1000, tension=100,
              density_profile='tapered', taper_ratio=10)

# Excite at thick end
whip.set_initial_pulse(position=10, amplitude=2.0)

# Solve and track tip velocity
solver = VerletSolver(whip)
solution = solver.solve(total_time=2.0, dt=0.001)

# Check for supersonic tip velocity
tip_velocities = solver.get_tip_velocity_history()
max_tip_velocity = np.max(tip_velocities)
sound_speed = 343  # m/s

print(f"Max tip velocity: {max_tip_velocity:.2f} m/s")
print(f"Supersonic: {max_tip_velocity > sound_speed}")
```

## Physical Validation

The simulation includes several validation checks:

1. **Energy Conservation**: Total energy should remain constant (within numerical error)
2. **CFL Condition**: $c \cdot dt / dx \leq 1$ for stability
3. **Boundary Conditions**: Properly enforced (fixed, free, or periodic)
4. **Wave Speed**: Matches theoretical $c = \sqrt{T/\mu}$

## Theory & Background

### Cosmological Connection (Advanced)

While this is primarily a classical simulation, the Lennard-Jones potential was originally inspired by hypothetical wave interactions in spacetime fabric. The constants can be related to cosmological parameters:

- $k_1 \approx 1.35 \times 10^{-66}$ derived from critical density balance
- $k_2$ can be calibrated to match dark energy equation of state

**This connection is speculative** and the simulation should primarily be understood as a classical wave mechanics tool.

---

## 🤖 Connections to Machine Learning Research

This simulation directly informs my approach to neural architecture design. Here are the key bridges:

### Wave Propagation → Information Flow

**Physics concept**: Wave speed varies with material properties: `c(x) = √(T/μ(x))`

**ML application**:
- Gradient flow in neural networks behaves like wave propagation
- Vanishing gradients = wave slowing down (high density)
- Exploding gradients = wave accelerating (low density, like whip tip)
- **Solution**: Skip connections (ResNets) act as "wave guides" for stable propagation

### Energy Conservation → Training Stability

**Physics concept**: Total energy `E = KE + PE` should be conserved (checked in `analysis.py`)

**ML application**:
- Monitor loss stability and gradient norms like physicists monitor energy
- CFL stability condition ↔ adaptive learning rates in optimizers (Adam, RMSprop)
- Energy drift detection ↔ training instability detection

### Energy Distribution → Attention Mechanisms

**Physics concept**: Energy density reveals where information concentrates

**ML application**:
- Attention weights as energy distribution across tokens
- Softmax normalization ↔ energy conservation (total attention = 1)
- **Leonardo Attention**: Energy-based attention mechanism inspired by this simulation

### Resonance Modes → Feature Learning

**Physics concept**: Systems develop stable resonance patterns

**ML application**:
- Neural networks learn feature representations (basis decomposition)
- Sparse features ↔ few active resonance modes (energy minimization)
- Feature importance ↔ resonance amplitude

### Numerical Integration → Optimization Algorithms

**Physics concept**: Multiple integrators (Central Difference, RK4, Verlet) with different trade-offs

**ML application**:
- SGD with Momentum ↔ Velocity Verlet (both use "velocity" term)
- Adaptive step size ↔ Adam optimizer (adjust "dt" based on local conditions)
- Energy conservation → symplectic integrators inspire stable optimization

### Observable Quantities → Interpretability

**Physics concept**: Measure observables (energy, momentum, wavelength) for understanding

**ML application**:
- Mechanistic interpretability through analogous observables
- Attention heatmaps ↔ energy density maps
- Gradient magnitudes ↔ energy flow
- Physics provides a template for what "interpretability" means

### How This Connects to My Research

- **QL-SIDeNN**: Self-interaction forces (`F(u)` in this simulation) inspire self-attention mechanisms
- **Leonardo Attention**: Energy minimization principles from this work inform attention weight computation
- **Bayesian Confidence Circuits**: Energy fluctuations and measurement uncertainty inform confidence quantification

**For detailed technical connections with code examples, see [ML_CONNECTIONS.md](ML_CONNECTIONS.md).**

---

## 🎓 What I Learned

### Physics Insights
- Wave-particle duality in computational contexts (discrete vs continuous)
- Conservation laws as optimization constraints
- Symmetry breaking and emergent patterns in dynamical systems

### Computational Insights
- Numerical stability challenges in wave equations (CFL condition, integrator choice)
- Trade-offs between accuracy, speed, and stability
- Importance of vectorization for performance (50-100x speedup)

### ML Connections
- Physics-inspired architectures offer **interpretability**
- Energy-based models provide **principled frameworks**
- Conservation laws enable **better inductive biases**
- Cross-disciplinary thinking leads to **novel architecture designs**

---

## Contributing

Contributions are welcome! Areas for improvement:
- 2D/3D wave simulations
- Additional boundary condition types
- Machine learning for parameter optimization
- Real-time interactive visualization with Streamlit

Please submit Pull Requests or open Issues for discussion.

## Performance Notes

- **Vectorization**: Uses NumPy arrays for 50-100x speedup vs Python loops
- **Memory**: Stores full time history - for long simulations, consider checkpointing
- **Integrators**: RK4 most accurate but ~4x slower than central difference

## Citation

If you use this simulation in academic work, please cite:
```
@software{wave_simulation_2025,
  author = {Dennis J. Carroll},
  title = {Classical Wave Dynamics Simulation with Whip Mechanics},
  year = {2025},
  url = {https://github.com/Dennis-J-Carroll/String-theory-Whip-force-Wave-force-Simulation-}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
