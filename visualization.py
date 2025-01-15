"""
Visualization functions for the whip force simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import constants as const
import solver

def plot_wave_evolution(u, x, dt, save_path=None):
    """
    Plot the wave displacement at different time steps.

    Parameters:
        u (numpy.ndarray): Wave displacement data
        x (numpy.ndarray): Spatial grid
        dt (float): Time step
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    for i in range(0, u.shape[0], 50):
        plt.plot(x, u[i, :], label=f't={i*dt:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Wave Propagation with Force Term')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_potential_energy(u_min=0.1, u_max=5, num_points=500, save_path=None):
    """
    Plot the potential energy function.

    Parameters:
        u_min (float): Minimum displacement value
        u_max (float): Maximum displacement value
        num_points (int): Number of points for plotting
        save_path (str, optional): Path to save the plot
    """
    u_values = np.linspace(u_min, u_max, num_points)
    potential_energy = solver.potential_function(u_values)

    plt.figure(figsize=(8, 6))
    plt.plot(u_values, potential_energy)
    plt.xlabel('u')
    plt.ylabel('V(u)')
    plt.title('Potential Energy Function')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_force_function(u_min=0.1, u_max=5, num_points=500, save_path=None):
    """
    Plot the force function.

    Parameters:
        u_min (float): Minimum displacement value
        u_max (float): Maximum displacement value
        num_points (int): Number of points for plotting
        save_path (str, optional): Path to save the plot
    """
    u_values = np.linspace(u_min, u_max, num_points)
    force = solver.force_function(u_values)

    plt.figure(figsize=(8, 6))
    plt.plot(u_values, force)
    plt.xlabel('u')
    plt.ylabel('F(u)')
    plt.title('Force Function')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()
