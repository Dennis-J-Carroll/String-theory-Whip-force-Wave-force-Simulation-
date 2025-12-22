"""
Interactive Demo Script for Wave Simulation

This script demonstrates the interactive visualization capabilities
of the wave simulation package. It generates several interactive
plots that can be explored in a web browser.

Usage:
    python interactive_demo.py
"""

import numpy as np
from solver import wave_solver, potential_function, force_function
from interactive_visualization import (
    create_animated_wave,
    create_3d_wave_surface,
    create_interactive_potential_force,
    create_energy_monitor,
    create_phase_space,
    create_dashboard_layout
)
from analysis import analyze_simulation, print_analysis_summary
import constants


def main():
    """
    Run interactive demo with multiple visualization types.
    """
    print("=" * 70)
    print(" 🌊 INTERACTIVE WAVE SIMULATION DEMO 🌊")
    print("=" * 70)
    print("\nInitializing simulation with default parameters...")
    print(f"  K1 = {constants.K1:.2e}")
    print(f"  K2 = {constants.K2:.2e}")
    print(f"  Wave Speed (c) = {constants.C}")
    print(f"  Time Step (dt) = {constants.DT}")
    print(f"  Space Step (dx) = {constants.DX}")
    print(f"  Total Time = {constants.TOTAL_TIME}")

    # Run simulation
    print("\n🔄 Running simulation...")
    u_history, time_points = wave_solver(
        constants.x,
        constants.u0,
        constants.u0_prev,
        constants.C,
        constants.DT,
        constants.TOTAL_TIME,
        constants.K1,
        constants.K2
    )
    print(f"✓ Simulation complete! {len(u_history)} time steps computed.\n")

    # Perform comprehensive analysis
    print("📊 Analyzing results...")
    analysis = analyze_simulation(
        u_history,
        time_points,
        constants.x,
        constants.DT,
        constants.DX,
        constants.K1,
        constants.K2
    )
    print_analysis_summary(analysis)

    # Generate interactive visualizations
    print("\n🎨 Generating interactive visualizations...")

    # 1. Animated 2D Wave
    print("  1. Creating animated wave propagation...")
    step = max(1, len(u_history) // 100)  # Sample for animation
    u_sampled = u_history[::step]
    t_sampled = time_points[::step]

    fig_animation = create_animated_wave(constants.x, u_sampled, t_sampled)
    fig_animation.write_html("output_animation.html")
    print("     Saved to: output_animation.html")

    # 2. 3D Surface Plot
    print("  2. Creating 3D wave evolution surface...")
    step_3d = max(1, len(u_history) // 50)  # Sample for 3D
    u_sampled_3d = u_history[::step_3d]
    t_sampled_3d = time_points[::step_3d]

    fig_3d = create_3d_wave_surface(constants.x, u_sampled_3d, t_sampled_3d)
    fig_3d.write_html("output_3d_surface.html")
    print("     Saved to: output_3d_surface.html")

    # 3. Potential and Force Functions
    print("  3. Creating potential & force function plot...")
    u_range = np.linspace(0.1, 5.0, 500)
    potential = potential_function(u_range, constants.K1, constants.K2)
    force = force_function(u_range, constants.K1, constants.K2)

    fig_potential = create_interactive_potential_force(u_range, potential, force)
    fig_potential.write_html("output_potential_force.html")
    print("     Saved to: output_potential_force.html")

    # 4. Energy Monitor
    print("  4. Creating energy conservation monitor...")
    fig_energy = create_energy_monitor(
        analysis['time_points'],
        analysis['energy']['kinetic'],
        analysis['energy']['potential'],
        analysis['energy']['total']
    )
    fig_energy.write_html("output_energy_monitor.html")
    print("     Saved to: output_energy_monitor.html")

    # 5. Phase Space
    print("  5. Creating phase space trajectory...")
    center_idx = len(constants.x) // 2
    fig_phase = create_phase_space(
        analysis['phase_space']['position'],
        analysis['phase_space']['velocity'],
        time_points,
        title=f"Phase Space at x={constants.x[center_idx]:.2f}"
    )
    fig_phase.write_html("output_phase_space.html")
    print("     Saved to: output_phase_space.html")

    # 6. Comprehensive Dashboard
    print("  6. Creating comprehensive dashboard...")
    fig_dashboard = create_dashboard_layout(
        constants.x,
        u_history,
        time_points,
        potential,
        force,
        u_range,
        analysis['energy']['kinetic'],
        analysis['energy']['potential']
    )
    fig_dashboard.write_html("output_dashboard.html")
    print("     Saved to: output_dashboard.html")

    print("\n" + "=" * 70)
    print(" ✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📂 Output files created:")
    print("  • output_animation.html       - Animated wave propagation")
    print("  • output_3d_surface.html      - 3D spacetime visualization")
    print("  • output_potential_force.html - Potential & force functions")
    print("  • output_energy_monitor.html  - Energy conservation tracking")
    print("  • output_phase_space.html     - Phase space trajectory")
    print("  • output_dashboard.html       - Comprehensive dashboard")
    print("\n💡 TIP: Open any HTML file in your web browser to explore!")
    print("\n🚀 For real-time interactive controls, run:")
    print("   python dashboard_app.py")
    print("   Then open http://localhost:8050 in your browser")
    print("\n📓 For Jupyter notebook experience, run:")
    print("   jupyter notebook interactive_notebook.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
