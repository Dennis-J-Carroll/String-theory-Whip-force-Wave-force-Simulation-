# String Theory Whip Force Wave Simulation

This project implements a numerical simulation of wave dynamics inspired by string theory concepts, particularly focusing on the interaction between waves in a hypothetical "fabric" of spacetime. The simulation explores the potential contribution of these wave interactions to phenomena like the accelerating expansion of the universe (dark energy).

## Features

- Modified wave equation solver incorporating force terms derived from potential functions
- Lennard-Jones-like potential for modeling wave interactions
- Finite difference method implementation with stability checks
- Energy conservation monitoring
- Visualization tools for wave propagation and potential energy

## Project Structure

```
.
├── constants.py      # Physical constants and parameters
├── solver.py         # Numerical solvers and core functions
├── visualization.py  # Plotting and visualization functions
├── main.py          # Main script to run simulations
└── tests/           # Unit tests
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

Run the main simulation:
```bash
python main.py
```

Run tests:
```bash
python -m pytest tests/
```

## Theory

The simulation is based on a modified wave equation that incorporates:
- A repulsive force term (k1/u^12) representing quantum effects at small scales
- An attractive force term (k2/u^6) modeling binding forces at larger scales
- Constants calibrated to match observed cosmological parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
