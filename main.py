"""
Main script to run the wave simulation with enhanced features.

This demonstrates the new OOP architecture with String and Solver classes.
"""
import numpy as np
import os
from string_model import String
from solver import CentralDifferenceSolver, RK4Solver, VerletSolver
import visualization
import analysis


def run_basic_simulation():
    """Run a basic wave simulation with default parameters."""
    print("=" * 70)
    print("BASIC WAVE SIMULATION")
    print("=" * 70)

    # Create string
    string = String(
        length=50.0,
        num_points=500,
        tension=50.0,
        density_profile="uniform",
        density_uniform=0.01,
    )

    # Set initial conditions
    string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    # Create solver
    solver = CentralDifferenceSolver(string, enable_force=True)

    # Solve (use smaller dt to satisfy CFL condition)
    print("\nSolving wave equation...")
    displacement_history = solver.solve(total_time=5.0, dt=0.001, save_interval=100, verbose=True)

    # Visualizations
    print("\nGenerating visualizations...")

    # Wave evolution
    visualization.plot_wave_evolution(
        displacement_history,
        string.x,
        time_history=solver.time_history,
        num_snapshots=10,
        save_path="output/wave_evolution.png",
    )

    # Energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history, solver.energy_history, plot=True, save_path="output/energy_conservation.png"
    )

    # Phase space
    analysis.plot_phase_space(
        solver.displacement_history, solver.velocity_history, node_index=250, save_path="output/phase_space.png"
    )

    # Heatmap
    analysis.plot_spacetime_heatmap(
        solver.displacement_history, string.x, solver.time_history, save_path="output/spacetime_heatmap.png"
    )

    # Print summary
    stability_stats = analysis.check_numerical_stability(solver.displacement_history)
    analysis.print_simulation_summary("Central Difference", energy_stats, stability_stats, 5.0, 500)


def run_whip_simulation():
    """Run a whip crack simulation with tapered density."""
    print("\n" + "=" * 70)
    print("WHIP CRACK SIMULATION (Tapered Density)")
    print("=" * 70)

    # Create tapered string (whip)
    whip = String(
        length=100.0,
        num_points=1000,
        tension=100.0,
        density_profile="tapered",
        density_base=0.02,
        density_tip=0.002,
        taper_exponent=1.5,
    )

    # Set initial pulse at thick end
    whip.set_initial_pulse(position=10.0, amplitude=2.0, width=3.0)

    # Plot density profile
    visualization.plot_density_profile(
        whip.x, whip.density, whip.wave_speed, save_path="output/whip_density_profile.png"
    )

    # Create solver - use Verlet for better energy conservation
    solver = VerletSolver(whip, enable_force=False)  # Disable external force for pure whip dynamics

    # Solve with smaller timestep for accuracy
    print("\nSolving whip dynamics...")
    displacement_history = solver.solve(total_time=2.0, dt=0.001, save_interval=20, verbose=True)

    # Visualizations
    print("\nGenerating visualizations...")

    # Wave evolution
    visualization.plot_wave_evolution(
        displacement_history,
        whip.x,
        time_history=solver.time_history,
        num_snapshots=10,
        save_path="output/whip_evolution.png",
    )

    # Tip velocity analysis
    tip_stats = analysis.plot_tip_velocity(
        solver.time_history, solver.velocity_history, sound_speed=343.0, save_path="output/whip_tip_velocity.png"
    )

    print(f"\n{'='*70}")
    print("WHIP TIP VELOCITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Max Tip Velocity: {tip_stats['max_tip_velocity']:.2f} m/s")
    print(f"Mach Number: {tip_stats['mach_number']:.3f}")
    print(f"Supersonic: {'YES' if tip_stats['is_supersonic'] else 'NO'}")
    print(f"Time of Max Velocity: {tip_stats['max_tip_velocity_time']:.4f} s")
    print(f"{'='*70}\n")

    # Energy conservation
    energy_stats = analysis.analyze_energy_conservation(
        solver.time_history, solver.energy_history, plot=True, save_path="output/whip_energy.png"
    )


def run_solver_comparison():
    """Compare different numerical integrators."""
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON")
    print("=" * 70)

    # Create string
    string_base = String(length=50.0, num_points=500, tension=50.0, density_profile="uniform")
    string_base.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

    solvers = {
        "Central Difference": CentralDifferenceSolver,
        "RK4": RK4Solver,
        "Verlet": VerletSolver,
    }

    results = {}

    for solver_name, SolverClass in solvers.items():
        print(f"\nRunning {solver_name}...")

        # Create fresh string for each solver
        string = String(length=50.0, num_points=500, tension=50.0, density_profile="uniform")
        string.set_initial_gaussian(center=25.0, width=5.0, amplitude=1.0)

        solver_instance = SolverClass(string, enable_force=True)
        # Use smaller dt to satisfy CFL condition
        displacement_history = solver_instance.solve(total_time=2.0, dt=0.001, save_interval=200, verbose=False)

        results[solver_name] = displacement_history

        # Quick energy analysis
        energy_stats = analysis.analyze_energy_conservation(
            solver_instance.time_history, solver_instance.energy_history, plot=False
        )
        print(f"  Energy drift: {energy_stats['relative_drift']*100:.3f}%")

    # Create comparison plot
    visualization.create_comparison_plot(results, string_base.x, time_index=-1, save_path="output/solver_comparison.png")


def main():
    """Main entry point - run all demonstrations."""
    # Create output directory
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    print("\n" + "=" * 70)
    print("WAVE SIMULATION DEMO")
    print("Classical String Mechanics & Whip Forces")
    print("=" * 70 + "\n")

    # Run demonstrations
    try:
        run_basic_simulation()
        run_whip_simulation()
        run_solver_comparison()

        # Generate potential and force plots
        print("\nGenerating potential and force function plots...")
        visualization.plot_potential_energy(save_path="output/potential_function.png")
        visualization.plot_force_function(save_path="output/force_function.png")

        print("\n" + "=" * 70)
        print("ALL SIMULATIONS COMPLETE!")
        print("=" * 70)
        print("\nOutput files saved to:")
        print("  - output/       : Plots and visualizations")
        print("  - docs/         : Documentation files")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
