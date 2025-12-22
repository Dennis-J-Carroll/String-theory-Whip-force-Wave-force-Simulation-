# Development Guide & Cheatsheet

## Moving Forward with Interactive Wave Simulation

This guide provides a roadmap for extending and enhancing the wave simulation project with additional interactive features and mathematical components.

---

## Quick Reference

### File Organization
```
Core Simulation:
  constants.py  → Parameters & physical constants
  solver.py     → Numerical PDE solver
  analysis.py   → Energy & phase space calculations

Visualization:
  visualization.py            → Static matplotlib plots
  interactive_visualization.py → Plotly interactive components

Interactive Apps:
  interactive_demo.py        → Generate HTML visualizations
  dashboard_app.py           → Web dashboard (Dash)
  interactive_notebook.ipynb → Jupyter with widgets
```

### Common Tasks Cheatsheet

#### 1. Running Different Modes
```bash
# Quick test with static plots
python main.py

# Generate all interactive HTML files
python interactive_demo.py

# Launch web dashboard
python dashboard_app.py
# Then open: http://localhost:8050

# Jupyter notebook
jupyter notebook interactive_notebook.ipynb
```

#### 2. Installing New Dependencies
```bash
# After adding to requirements.txt
pip install -r requirements.txt

# For development
pip install -e .
```

#### 3. Running Tests
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_solver.py

# With coverage
pytest --cov=. tests/
```

---

## Enhancement Roadmap

### Phase 1: Current Features (COMPLETED ✓)

- [x] Animated 2D wave propagation with Plotly
- [x] 3D spacetime surface visualization
- [x] Energy conservation monitoring
- [x] Phase space analysis
- [x] Interactive potential/force plots
- [x] Web dashboard with parameter controls
- [x] Jupyter notebook interface

### Phase 2: Near-Term Enhancements (Next Steps)

#### A. Enhanced Physics
```python
# File: solver.py or new physics_models.py

1. Multiple Wave Interactions
   - Add support for multiple concurrent waves
   - Wave superposition and interference
   - Collision dynamics

2. Variable Wave Speed
   - Make c(x) position-dependent
   - Model different "media" or "field densities"

3. Damping/Dissipation
   - Add friction term: -γ(∂u/∂t)
   - Energy dissipation analysis

4. External Forcing
   - Periodic driving force
   - Impulse forcing
   - Stochastic forcing (noise)
```

#### B. Advanced Visualizations
```python
# File: interactive_visualization.py

1. Heatmap/Density Plots
   def create_wave_heatmap(x, time_points, u_history):
       # Spacetime heatmap showing wave intensity

2. Fourier Analysis Display
   def create_fourier_spectrum_animation(u_history, dx):
       # Show how frequency content evolves

3. Wavelet Analysis
   def create_wavelet_transform(u, dt):
       # Time-frequency analysis

4. Vector Field Visualization
   def create_velocity_field(u_history, dt):
       # Arrow plots showing wave velocity field
```

#### C. Comparative Analysis Tools
```python
# File: new file - comparison.py

1. Parameter Sweep Visualization
   def parameter_sweep(param_name, param_range):
       # Run simulations across parameter range
       # Visualize outcomes as heatmap

2. Bifurcation Diagrams
   def bifurcation_analysis(control_param):
       # Identify regime changes

3. Sensitivity Analysis
   def sensitivity_matrix():
       # How outputs depend on each input
```

### Phase 3: Advanced Features (Future)

#### A. 2D Spatial Domain
```python
# Extend to 2D waves
# Modified solver: u(x, y, t)

def wave_solver_2d(x_grid, y_grid, u0, ...):
    # 2D wave equation
    # Visualize as animated surface or contour plots
```

#### B. Machine Learning Integration
```python
# File: ml_analysis.py

1. Parameter Optimization
   - Use ML to find optimal K1, K2 for desired behavior
   - Neural network to predict wave evolution

2. Anomaly Detection
   - Identify unusual wave behaviors
   - Classify wave patterns
```

#### C. Real-Time Performance
```python
1. GPU Acceleration
   - Use CuPy or JAX for GPU computation
   - Massive speedup for large grids

2. Parallelization
   - Multi-core processing with multiprocessing
   - Distributed computing with Dask
```

#### D. Advanced Interactivity
```python
# File: enhanced_dashboard.py

1. Drawing Tools
   - Let users draw initial conditions with mouse
   - Click to add wave sources

2. Real-Time Simulation
   - Stream results as they compute
   - WebSocket updates

3. Collaborative Features
   - Share simulations via URL
   - Save/load configurations
```

---

## Code Patterns & Recipes

### Pattern 1: Adding a New Visualization

```python
# Step 1: Add function to interactive_visualization.py
def create_my_new_plot(data, title="My Plot"):
    """
    Your visualization function.

    Args:
        data: Your data structure
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add your traces
    fig.add_trace(go.Scatter(...))

    # Configure layout
    fig.update_layout(
        title=title,
        template='plotly_white'
    )

    return fig

# Step 2: Add to interactive_demo.py
fig_new = create_my_new_plot(data)
fig_new.write_html("output_my_plot.html")

# Step 3: Add tab to dashboard_app.py
dbc.Tab(label="My Plot", tab_id="tab-myplot")

# Step 4: Add callback handler
elif active_tab == "tab-myplot":
    fig = create_my_new_plot(sim_data)
    return dcc.Graph(figure=fig)
```

### Pattern 2: Adding a New Physics Model

```python
# Step 1: Define in solver.py or new module
def alternative_force_function(u, params):
    """
    New force model: F(u) = ...
    """
    # Your physics here
    return force

# Step 2: Add to wave_solver
def wave_solver(..., force_function=force_function):
    # Make force function selectable
    F = force_function(u, params)

# Step 3: Add to dashboard controls
dcc.Dropdown(
    id='force-model',
    options=[
        {'label': 'Lennard-Jones', 'value': 'lj'},
        {'label': 'My New Model', 'value': 'new'}
    ]
)
```

### Pattern 3: Adding Analysis Features

```python
# File: analysis.py

def my_new_analysis(u_history, x, time_points):
    """
    Compute interesting property.

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Your analysis computations
    results['property1'] = ...
    results['property2'] = ...

    return results

# Then use in interactive_demo.py
analysis = analyze_simulation(...)
new_results = my_new_analysis(u_history, x, time_points)
```

---

## Technology Stack Reference

### Core Scientific Computing
- **NumPy**: Array operations, linear algebra
- **SciPy**: Advanced numerical methods (optional, for future)
- **SymPy**: Symbolic mathematics (optional, for derivations)

### Visualization
- **Plotly**: Interactive 2D/3D plots
- **Matplotlib**: Static publication-quality plots
- **Dash**: Web dashboard framework
- **ipywidgets**: Jupyter notebook widgets

### Notebook & Web
- **Jupyter**: Interactive notebooks
- **Dash Bootstrap Components**: UI components for dashboard
- **Kaleido**: Export Plotly figures to static images

### Testing
- **pytest**: Unit testing framework
- **pytest-cov**: Code coverage

---

## Extension Ideas by Difficulty

### Easy (1-2 hours each)
1. Add new color schemes to visualizations
2. Export animations as GIF or video
3. Add download button for simulation data (CSV)
4. Create preset parameter configurations
5. Add tooltips explaining each parameter

### Medium (Half day each)
1. Implement wave packet analysis (width, center tracking)
2. Add multiple initial condition presets (soliton, pulse, etc.)
3. Create comparison mode (run 2 simulations side-by-side)
4. Add Fourier transform visualization
5. Implement automatic parameter optimization

### Advanced (1-2 days each)
1. Extend to 2D spatial domain
2. Add absorbing boundary conditions
3. Implement GPU acceleration with CuPy
4. Create ML model to predict stability
5. Add real-time collaborative features

### Research-Level (Week+ each)
1. Full 3D simulation (x, y, z, t)
2. Quantum corrections to wave equation
3. Relativistic wave propagation
4. Coupling to gravitational field
5. Monte Carlo uncertainty quantification

---

## Best Practices

### Code Organization
```python
# Keep modules focused:
# - solver.py: ONLY numerical methods
# - analysis.py: ONLY analysis functions
# - visualization.py: ONLY plotting
# - constants.py: ONLY parameters

# Use clear naming:
# - Functions: verb_noun (calculate_energy, create_plot)
# - Variables: descriptive (kinetic_energy, not ke)
# - Constants: UPPER_CASE
```

### Documentation
```python
def your_function(arg1, arg2):
    """
    One-line summary.

    Longer description if needed. Explain the physics,
    the numerical method, or the visualization strategy.

    Args:
        arg1: Description with units
        arg2: Description with type info

    Returns:
        Description of return value with shape/type

    Example:
        >>> result = your_function(1.0, 2.0)
        >>> print(result)
    """
```

### Testing
```python
# tests/test_new_feature.py

def test_my_function():
    # Arrange
    input_data = np.array([1, 2, 3])

    # Act
    result = my_function(input_data)

    # Assert
    expected = np.array([2, 4, 6])
    np.testing.assert_array_almost_equal(result, expected)
```

### Git Workflow
```bash
# Feature branches
git checkout -b feature/fourier-analysis
# ... make changes ...
git add .
git commit -m "Add Fourier transform visualization"
git push origin feature/fourier-analysis

# Create PR when ready
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Performance Issues
```python
# BAD: Using Python loops
for i in range(len(x)):
    u[i] = some_calculation(x[i])

# GOOD: Use NumPy vectorization
u = some_calculation(x)
```

### Pitfall 2: Memory Issues with Large Grids
```python
# Solution: Downsample for visualization
step = max(1, len(u_history) // 100)
u_sampled = u_history[::step]
```

### Pitfall 3: Unstable Simulations
```python
# Always check CFL condition
from solver import check_cfl_condition

is_stable, cfl = check_cfl_condition(c, dt, dx)
if not is_stable:
    print(f"Warning: CFL={cfl:.3f} > 1, reduce dt or increase dx")
```

### Pitfall 4: Division by Zero in Potential
```python
# Always add small offset
u_safe = np.abs(u) + 0.1  # Prevent singularity
V = potential_function(u_safe, K1, K2)
```

---

## Resources for Learning

### Numerical Methods
- "Numerical Recipes" - Press et al.
- "Finite Difference Methods for PDEs" - LeVeque
- Online: SciPy lectures (scipy-lectures.org)

### Visualization
- Plotly documentation: plotly.com/python
- Dash tutorials: dash.plotly.com
- "Interactive Data Visualization" - Ward et al.

### Wave Physics
- "Introduction to Wave Phenomena" - Towne
- "Nonlinear Waves" - Infeld & Rowlands
- Khan Academy: Physics - Waves

### Python Best Practices
- PEP 8 style guide
- "Fluent Python" - Ramalho
- Real Python tutorials

---

## Quick Tips

### Debugging Visualization Issues
```python
# Print figure structure
print(fig)

# Save figure data
fig.write_json("debug_figure.json")

# Check data ranges
print(f"Data range: {np.min(data)} to {np.max(data)}")
```

### Performance Profiling
```python
# Time a function
import time
start = time.time()
result = slow_function()
print(f"Took {time.time() - start:.3f} seconds")

# Profile with cProfile
python -m cProfile -o profile.stats interactive_demo.py
```

### Interactive Development
```python
# Use IPython for experimentation
ipython

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Then import and test
from solver import wave_solver
```

---

## Next Steps Checklist

Immediate (Do First):
- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Run basic demo: `python interactive_demo.py`
- [ ] Launch dashboard: `python dashboard_app.py`
- [ ] Explore Jupyter notebook
- [ ] Run test suite to ensure everything works

Short-Term (This Week):
- [ ] Add a new preset initial condition
- [ ] Create a custom color scheme
- [ ] Export a simulation as video/GIF
- [ ] Add Fourier analysis visualization
- [ ] Write a test for new feature

Medium-Term (This Month):
- [ ] Implement 2D wave simulation
- [ ] Add ML-based parameter optimization
- [ ] Create collaborative features
- [ ] Performance optimization with Numba/Cython
- [ ] Publication-quality figure generation

Long-Term (This Quarter):
- [ ] Full 3D simulation
- [ ] GPU acceleration
- [ ] Real-time streaming visualization
- [ ] Integration with external data sources
- [ ] Mobile-friendly dashboard

---

## Getting Help

1. **Documentation**: Check docstrings in code
2. **Examples**: Look at existing functions for patterns
3. **Tests**: tests/ directory shows usage examples
4. **Issues**: Check GitHub issues for similar problems
5. **Community**:
   - Plotly community forum
   - Scientific Python Discourse
   - Stack Overflow (tags: plotly, dash, numerical-methods)

---

## Contributing

When adding new features:

1. **Branch**: Create feature branch
2. **Develop**: Write code + tests
3. **Document**: Add docstrings + update README
4. **Test**: Run pytest, ensure coverage
5. **PR**: Create pull request with description

Code review checklist:
- [ ] Follows PEP 8 style
- [ ] Has docstrings
- [ ] Has unit tests
- [ ] Passes all tests
- [ ] Documentation updated
- [ ] No performance regressions

---

## Conclusion

This project now has a solid foundation for interactive mathematical visualization. The modular architecture makes it easy to:

- Add new physics models (modify solver.py)
- Create new visualizations (extend interactive_visualization.py)
- Build new interfaces (create new dashboard/notebook)
- Perform new analyses (extend analysis.py)

**Key Philosophy**:
- Keep core simulation clean and fast
- Make visualization rich and interactive
- Separate concerns (physics ≠ visualization ≠ UI)
- Test everything
- Document clearly

**Remember**: Start simple, iterate quickly, and always validate against known results!

---

*Happy Simulating!* 🌊✨
