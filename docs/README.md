# Documentation

This folder contains generated documentation and visualizations.

## Contents

After running simulations, this folder will contain:

- **Wave animations** (`.gif` files)
- **Energy plots** (`.png` files)
- **Phase space diagrams**
- **Space-time heatmaps**
- **Whip dynamics visualizations**

## Generating Documentation

Run the main simulation to generate all visualizations:

```bash
python main.py
```

This will populate both the `output/` and `docs/` directories with results.

## Example Outputs

1. **wave_animation.gif** - Animated wave propagation
2. **energy_conservation.png** - Energy vs time plot
3. **phase_space.png** - Phase space trajectories
4. **spacetime_heatmap.png** - Wave evolution heatmap
5. **whip_tip_velocity.png** - Whip crack velocity analysis

## Notes

- All visualization files are git-ignored by default
- Run simulations locally to generate fresh outputs
- Animations require `matplotlib` and `pillow` packages
