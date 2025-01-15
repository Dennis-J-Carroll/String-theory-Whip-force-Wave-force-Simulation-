"""
Main script to run the whip force simulation.
"""
import constants as const
import solver
import visualization

def main():
    # Create initial conditions
    u0 = const.create_initial_conditions()

    # Solve the wave equation
    u = solver.wave_equation_solver(u0, const.DT, const.DX, const.T)

    # Create visualizations
    visualization.plot_wave_evolution(u, const.X, const.DT)
    visualization.plot_potential_energy()
    visualization.plot_force_function()

if __name__ == "__main__":
    main()
