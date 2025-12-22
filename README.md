# Wave Dynamics Simulation: Classical String Mechanics & Wave Forces

![Wave Simulation](docs/wave_animation.gif)
*Note: Run simulation to generate visualization*

## Overview

This project implements a comprehensive numerical simulation of **classical wave mechanics** on strings with variable properties. Despite the repository name referencing "String Theory," this is a **classical physics simulation** focusing on:

- **Wave propagation** on strings with spatially-varying properties
- **Whip mechanics** - modeling the famous "whip crack" supersonic tip phenomenon
- **Modified wave equations** with Lennard-Jones-like potentials for wave interactions
- **Energy conservation** in non-linear wave systems

### !! Important Clarification: Classical vs. Quantum

**This is NOT quantum string theory** (the fundamental physics theory proposing 1-dimensional strings as basic constituents of matter). This simulation models **classical continuous strings** (like ropes, cables, or whips) governed by Newtonian mechanics and the classical wave equation. This is at the very least the practice of programming an idea with the pure intent to imaginatuvely experiment. 

## Features

### Physics Engine
- ✅ Multiple numerical integrators: Central Difference, RK4, Velocity Verlet
- ✅ Vectorized NumPy implementation for performance
- ✅ Lennard-Jones-like potential for modeling wave interactions
- ✅ Linear density tapering for whip simulations
- ✅ CFL stability condition checking
- ✅ Real-time energy conservation monitoring

### Visualization & Analysis
- Wave evolution plots with customizable time steps
- Energy conservation plots (Kinetic, Potential, Total)
- Phase space trajectory visualization
- Space-time heatmaps for wave history
- Animated GIF generation
- Potential and force function plots

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
├── string_model.py      # String class for wave properties
├── solver.py            # Numerical solvers (Central Diff, RK4, Verlet)
├── visualization.py     # Plotting and visualization functions
├── analysis.py          # Energy tracking and phase space analysis
├── main.py              # Main simulation script
├── tests/               # Unit tests
│   ├── test_solver.py
│   ├── test_string.py
│   └── test_energy.py
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
