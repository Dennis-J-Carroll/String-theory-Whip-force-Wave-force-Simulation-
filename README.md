# String Theory Whip Force Wave Simulation

This project implements a numerical simulation of wave dynamics inspired by string theory concepts, particularly focusing on the interaction between waves in a hypothetical "fabric" of spacetime. The simulation explores the potential contribution of these wave interactions to phenomena like the accelerating expansion of the universe (dark energy).

## Features

### Core Simulation
- Modified wave equation solver incorporating force terms derived from potential functions
- Lennard-Jones-like potential for modeling wave interactions
- Finite difference method implementation with stability checks (CFL condition)
- Energy conservation monitoring and analysis
- Phase space trajectory analysis

### Interactive Visualizations
- **Animated 2D Wave Propagation** - Watch waves evolve with play/pause controls and time slider
- **3D Spacetime Surface** - Explore wave evolution in 3D with interactive rotation and zoom
- **Energy Conservation Monitor** - Real-time tracking of kinetic, potential, and total energy
- **Phase Space Visualization** - Analyze system dynamics in position-velocity space
- **Interactive Potential & Force Functions** - Dual-axis plots with hover interactions
- **Web-Based Dashboard** - Full-featured web app with real-time parameter controls
- **Jupyter Notebook Interface** - Interactive widgets for parameter exploration

## Project Structure

```
.
├── constants.py                # Physical constants and parameters
├── solver.py                   # Numerical solvers and core functions
├── visualization.py            # Static matplotlib visualizations
├── interactive_visualization.py # Plotly-based interactive visualizations
├── analysis.py                 # Energy and phase space analysis tools
├── main.py                     # Basic simulation runner (matplotlib)
├── interactive_demo.py         # Interactive demo (generates HTML files)
├── dashboard_app.py            # Web-based dashboard (Dash app)
├── interactive_notebook.ipynb  # Jupyter notebook with widgets
├── whip_force_Sim.py          # Standalone variant
├── whip_sim_02.py             # Enhanced standalone variant
└── tests/                     # Unit tests
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

### Option 1: Static Visualization (Quick Start)
Run the basic simulation with matplotlib plots:
```bash
python main.py
```

### Option 2: Interactive HTML Visualizations
Generate interactive HTML files that can be opened in any web browser:
```bash
python interactive_demo.py
```

This creates six HTML files:
- `output_animation.html` - Animated wave with play/pause controls
- `output_3d_surface.html` - 3D spacetime visualization
- `output_potential_force.html` - Interactive potential & force plots
- `output_energy_monitor.html` - Energy conservation tracking
- `output_phase_space.html` - Phase space trajectory
- `output_dashboard.html` - Comprehensive multi-plot dashboard

### Option 3: Web Dashboard (Real-Time Controls)
Launch an interactive web application with live parameter controls:
```bash
python dashboard_app.py
```

Then open your browser to: `http://localhost:8050`

Features:
- Real-time parameter sliders (K1, K2, wave speed, resolution, etc.)
- Instant visualization updates
- Multiple tabs for different analysis views
- No need to restart - adjust and re-run on the fly

### Option 4: Jupyter Notebook (Educational/Exploratory)
For an interactive notebook experience with widgets:
```bash
jupyter notebook interactive_notebook.ipynb
```

Features:
- Interactive sliders and controls embedded in notebook
- Step-by-step exploration of physics
- Compare multiple scenarios side-by-side
- Perfect for teaching and learning

### Running Tests
```bash
python -m pytest tests/
```

## Interactive Features

### Parameter Controls
Adjust simulation parameters in real-time:
- **Force Constants**: K1 (repulsive) and K2 (attractive) on logarithmic scales
- **Wave Properties**: Wave speed, initial amplitude, position, width
- **Numerical Settings**: Time step (dt), spatial step (dx), total simulation time
- **Stability Monitoring**: Automatic CFL condition checking

### Visualization Types

#### 1. Animated Wave Propagation
- Play/pause animation controls
- Time slider to scrub through simulation
- Hover to see exact values
- Smooth transitions between frames

#### 2. 3D Spacetime Surface
- Full 3D visualization of wave evolution
- Interactive rotation, zoom, and pan
- Contour projections for better depth perception
- Customizable camera angles

#### 3. Energy Conservation Analysis
- Kinetic energy (motion)
- Potential energy (field)
- Total energy tracking
- Conservation quality metrics

#### 4. Phase Space Explorer
- Position vs velocity trajectories
- Color-coded by time
- Identify periodic/chaotic behavior
- Origin marker for reference

#### 5. Multi-Scenario Comparison
- Run multiple simulations side-by-side
- Compare different initial conditions
- Statistical analysis of outcomes
- Identify parameter sensitivities

## Theory

The simulation is based on a modified wave equation that incorporates:
- A repulsive force term (K1/u^12) representing quantum effects at small scales
- An attractive force term (K2/u^6) modeling binding forces at larger scales
- Constants calibrated to match observed cosmological parameters (critical density ρ_c)

### Modified Wave Equation
```
∂²u/∂t² = c² ∂²u/∂x² + F(u)
```

Where:
- `u(x,t)` is the wave displacement
- `c` is the wave propagation speed
- `F(u) = 12·K1/u¹³ - 6·K2/u⁷` is the force derived from the potential
- `V(u) = K1/u¹² - K2/u⁶` is the Lennard-Jones-like potential

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
