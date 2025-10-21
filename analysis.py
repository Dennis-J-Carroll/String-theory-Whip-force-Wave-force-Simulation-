"""
Analysis module for wave simulations.

This module provides tools for:
- Energy calculations (kinetic, potential, total)
- Phase space analysis
- Statistical analysis of wave properties
- Conservation checks
"""

import numpy as np
from typing import List, Tuple, Dict
from solver import potential_function


def calculate_kinetic_energy(u_history: List[np.ndarray],
                             u_prev_history: List[np.ndarray],
                             dt: float,
                             dx: float) -> np.ndarray:
    """
    Calculate kinetic energy at each time step.

    The kinetic energy density is (1/2) * (∂u/∂t)².
    We approximate the time derivative using finite differences.

    Args:
        u_history: List of displacement arrays
        u_prev_history: List of previous displacement arrays
        dt: Time step
        dx: Spatial step

    Returns:
        Array of kinetic energy values at each time step
    """
    kinetic_energy = []

    for i in range(len(u_history)):
        if i == 0:
            # For first step, use zero velocity
            velocity = np.zeros_like(u_history[0])
        else:
            # Approximate velocity using finite differences
            velocity = (u_history[i] - u_history[i-1]) / dt

        # Kinetic energy: integrate (1/2) * v² over space
        ke = 0.5 * np.sum(velocity**2) * dx
        kinetic_energy.append(ke)

    return np.array(kinetic_energy)


def calculate_potential_energy(u_history: List[np.ndarray],
                               K1: float,
                               K2: float,
                               dx: float,
                               u_offset: float = 0.1) -> np.ndarray:
    """
    Calculate potential energy at each time step.

    Args:
        u_history: List of displacement arrays
        K1: Repulsive force constant
        K2: Attractive force constant
        dx: Spatial step
        u_offset: Small offset to avoid division by zero

    Returns:
        Array of potential energy values at each time step
    """
    potential_energy = []

    for u in u_history:
        # Add small offset to avoid singularities
        u_safe = np.abs(u) + u_offset

        # Calculate potential at each point
        V = potential_function(u_safe, K1, K2)

        # Integrate over space
        pe = np.sum(V) * dx
        potential_energy.append(pe)

    return np.array(potential_energy)


def calculate_total_energy(kinetic_energy: np.ndarray,
                          potential_energy: np.ndarray) -> np.ndarray:
    """
    Calculate total energy (kinetic + potential).

    Args:
        kinetic_energy: Array of kinetic energy values
        potential_energy: Array of potential energy values

    Returns:
        Array of total energy values
    """
    return kinetic_energy + potential_energy


def check_energy_conservation(total_energy: np.ndarray,
                              tolerance: float = 0.05) -> Dict[str, float]:
    """
    Check if energy is conserved within a given tolerance.

    Args:
        total_energy: Array of total energy values
        tolerance: Acceptable relative change in energy (default 5%)

    Returns:
        Dictionary with conservation statistics
    """
    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    max_energy = np.max(total_energy)
    min_energy = np.min(total_energy)

    # Calculate relative changes
    relative_change = abs(final_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
    max_deviation = max(
        abs(max_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0,
        abs(min_energy - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
    )

    is_conserved = relative_change < tolerance

    return {
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'relative_change': relative_change,
        'max_deviation': max_deviation,
        'is_conserved': is_conserved,
        'tolerance': tolerance
    }


def calculate_phase_space_trajectory(u_history: List[np.ndarray],
                                     dt: float,
                                     spatial_index: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate phase space trajectory (position vs velocity) for a specific spatial point.

    Args:
        u_history: List of displacement arrays
        dt: Time step
        spatial_index: Index of spatial point to analyze (default: center point)

    Returns:
        Tuple of (position, velocity) arrays
    """
    if spatial_index is None:
        spatial_index = len(u_history[0]) // 2

    # Extract position at the chosen spatial point over time
    position = np.array([u[spatial_index] for u in u_history])

    # Calculate velocity using finite differences
    velocity = np.gradient(position, dt)

    return position, velocity


def calculate_wave_statistics(u_history: List[np.ndarray],
                             time_points: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate statistical properties of the wave over time.

    Args:
        u_history: List of displacement arrays
        time_points: Array of time values

    Returns:
        Dictionary with various statistical measures
    """
    stats = {
        'mean': [],
        'std': [],
        'max': [],
        'min': [],
        'energy_density': []
    }

    for u in u_history:
        stats['mean'].append(np.mean(u))
        stats['std'].append(np.std(u))
        stats['max'].append(np.max(u))
        stats['min'].append(np.min(u))
        stats['energy_density'].append(np.sum(u**2))

    # Convert to numpy arrays
    for key in stats:
        stats[key] = np.array(stats[key])

    return stats


def find_wave_peaks(u: np.ndarray, x: np.ndarray,
                   threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in the wave displacement.

    Args:
        u: Displacement array
        x: Spatial grid
        threshold: Minimum height for peak detection

    Returns:
        Tuple of (peak_positions, peak_heights)
    """
    # Find local maxima
    peaks = []
    positions = []

    for i in range(1, len(u) - 1):
        if u[i] > u[i-1] and u[i] > u[i+1] and u[i] > threshold:
            peaks.append(u[i])
            positions.append(x[i])

    return np.array(positions), np.array(peaks)


def calculate_wave_velocity(u_history: List[np.ndarray],
                           x: np.ndarray,
                           time_points: np.ndarray,
                           threshold: float = 0.5) -> float:
    """
    Estimate wave propagation velocity by tracking peak movement.

    Args:
        u_history: List of displacement arrays
        x: Spatial grid
        time_points: Time values
        threshold: Threshold for peak detection

    Returns:
        Estimated wave velocity
    """
    peak_positions = []
    peak_times = []

    for i, (u, t) in enumerate(zip(u_history, time_points)):
        positions, heights = find_wave_peaks(u, x, threshold)
        if len(positions) > 0:
            # Use the highest peak
            max_idx = np.argmax(heights)
            peak_positions.append(positions[max_idx])
            peak_times.append(t)

    if len(peak_positions) < 2:
        return 0.0

    # Fit a line to peak positions vs time
    peak_positions = np.array(peak_positions)
    peak_times = np.array(peak_times)

    # Simple linear regression
    velocity = np.polyfit(peak_times, peak_positions, 1)[0]

    return velocity


def calculate_fourier_spectrum(u: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the spatial Fourier spectrum of the wave.

    Args:
        u: Displacement array
        dx: Spatial step

    Returns:
        Tuple of (wavenumbers, power_spectrum)
    """
    # Compute FFT
    spectrum = np.fft.fft(u)
    power = np.abs(spectrum)**2

    # Compute wavenumbers
    N = len(u)
    k = 2 * np.pi * np.fft.fftfreq(N, dx)

    # Only return positive frequencies
    positive_idx = k >= 0
    k = k[positive_idx]
    power = power[positive_idx]

    return k, power


def analyze_simulation(u_history: List[np.ndarray],
                      time_points: np.ndarray,
                      x: np.ndarray,
                      dt: float,
                      dx: float,
                      K1: float,
                      K2: float) -> Dict:
    """
    Perform comprehensive analysis of a wave simulation.

    Args:
        u_history: List of displacement arrays
        time_points: Time values
        x: Spatial grid
        dt: Time step
        dx: Spatial step
        K1: Repulsive force constant
        K2: Attractive force constant

    Returns:
        Dictionary containing all analysis results
    """
    # Energy calculations
    u_prev_history = [u_history[0]] + u_history[:-1]
    kinetic_energy = calculate_kinetic_energy(u_history, u_prev_history, dt, dx)
    potential_energy = calculate_potential_energy(u_history, K1, K2, dx)
    total_energy = calculate_total_energy(kinetic_energy, potential_energy)

    # Energy conservation check
    conservation = check_energy_conservation(total_energy)

    # Phase space
    position, velocity = calculate_phase_space_trajectory(u_history, dt)

    # Wave statistics
    stats = calculate_wave_statistics(u_history, time_points)

    # Wave velocity
    wave_velocity = calculate_wave_velocity(u_history, x, time_points)

    # Fourier spectrum at final time
    k, power = calculate_fourier_spectrum(u_history[-1], dx)

    results = {
        'energy': {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': total_energy,
            'conservation': conservation
        },
        'phase_space': {
            'position': position,
            'velocity': velocity
        },
        'statistics': stats,
        'wave_velocity': wave_velocity,
        'fourier': {
            'wavenumbers': k,
            'power_spectrum': power
        },
        'time_points': time_points,
        'x': x
    }

    return results


def print_analysis_summary(analysis: Dict):
    """
    Print a formatted summary of the analysis results.

    Args:
        analysis: Dictionary from analyze_simulation()
    """
    print("=" * 60)
    print("WAVE SIMULATION ANALYSIS SUMMARY")
    print("=" * 60)

    # Energy conservation
    cons = analysis['energy']['conservation']
    print(f"\nEnergy Conservation:")
    print(f"  Initial Energy: {cons['initial_energy']:.6e}")
    print(f"  Final Energy:   {cons['final_energy']:.6e}")
    print(f"  Relative Change: {cons['relative_change']*100:.2f}%")
    print(f"  Max Deviation:   {cons['max_deviation']*100:.2f}%")
    print(f"  Conserved (within {cons['tolerance']*100:.0f}%): {cons['is_conserved']}")

    # Wave velocity
    print(f"\nWave Propagation:")
    print(f"  Estimated Velocity: {analysis['wave_velocity']:.4f}")

    # Statistics at final time
    stats = analysis['statistics']
    print(f"\nFinal State Statistics:")
    print(f"  Mean Displacement: {stats['mean'][-1]:.6e}")
    print(f"  Std Deviation:     {stats['std'][-1]:.6e}")
    print(f"  Max Displacement:  {stats['max'][-1]:.6e}")
    print(f"  Min Displacement:  {stats['min'][-1]:.6e}")

    print("=" * 60)
