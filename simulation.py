import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
N_ATOMS = 200
SIMULATION_TIME = 4e-3  # seconds
DT = 1e-6  # seconds
BOX_SIZE = 1e-3 # meters

# --- Physical Constants ---
ATOMIC_MASS = 87 * 1.66e-27  # Mass of Rubidium-87 in kg
K_BOLTZMANN = 1.38e-23  # Boltzmann's constant
HBAR = 1.054571817e-34  # Reduced Planck constant

# --- Laser Parameters ---
LASER_WAVELENGTH = 780e-9  # meters
LASER_DETUNING = -10e6  # Hz (negative for red-detuned)
LASER_INTENSITY = 1.0  # Saturation intensity units (peak)
LASER_BEAM_WAIST = 5e-4  # meters

# --- MOT Parameters ---
MOT_TRAP_STRENGTH = 3e-17  # N/m

def initialize_atoms(n_atoms, initial_temp, box_size):
    """
    Initializes the positions and velocities of atoms.
    """
    # Initial positions (randomly distributed in a box)
    positions = (np.random.rand(n_atoms, 2) - 0.5) * box_size

    # Initial velocities (from Maxwell-Boltzmann distribution)
    v_rms = np.sqrt(2 * K_BOLTZMANN * initial_temp / ATOMIC_MASS)
    velocities = np.random.normal(0, v_rms, (n_atoms, 2))
    
    return positions, velocities

def calculate_doppler_force(positions, velocities, laser_detuning, laser_intensity, k, beam_waist):
    """
    Calculates the velocity-dependent Doppler cooling force and the total scattering rate
    for Gaussian laser beams.
    """
    # --- Constants ---
    GAMMA = 2 * np.pi * 6.065e6  # Natural linewidth for Rb-87 D2 line in rad/s
    
    # --- Laser parameters ---
    delta = 2 * np.pi * laser_detuning
    s0_peak = laser_intensity
    w = beam_waist

    # --- Position and velocity components ---
    x = positions[:, 0]
    y = positions[:, 1]
    vx = velocities[:, 0]
    vy = velocities[:, 1]

    # --- Position-dependent saturation parameters ---
    s0_x_beams = s0_peak * np.exp(-2 * y**2 / w**2)
    s0_y_beams = s0_peak * np.exp(-2 * x**2 / w**2)

    # --- Force and scattering rate calculation ---
    # For +x beam
    lorentzian_plus_x = (s0_x_beams) / (1 + s0_x_beams + (2 * (delta - k * vx) / GAMMA)**2)
    # For -x beam
    lorentzian_minus_x = (s0_x_beams) / (1 + s0_x_beams + (2 * (delta + k * vx) / GAMMA)**2)
    # For +y beam
    lorentzian_plus_y = (s0_y_beams) / (1 + s0_y_beams + (2 * (delta - k * vy) / GAMMA)**2)
    # For -y beam
    lorentzian_minus_y = (s0_y_beams) / (1 + s0_y_beams + (2 * (delta + k * vy) / GAMMA)**2)

    # Total scattering rate
    total_scattering_rate = (GAMMA / 2) * (lorentzian_plus_x + lorentzian_minus_x + lorentzian_plus_y + lorentzian_minus_y)

    # Force
    force_x = HBAR * k * (GAMMA / 2) * (lorentzian_plus_x - lorentzian_minus_x)
    force_y = HBAR * k * (GAMMA / 2) * (lorentzian_plus_y - lorentzian_minus_y)
    
    return np.column_stack((force_x, force_y)), total_scattering_rate

def main():
    """
    Main simulation function.
    """
    # Initialization
    initial_temp = 50e-3  # 50mK
    positions, velocities = initialize_atoms(N_ATOMS, initial_temp, BOX_SIZE)
    
    # Laser wave vector
    k = 2 * np.pi / LASER_WAVELENGTH

    n_steps = int(SIMULATION_TIME / DT)
    
    # Store initial velocities for comparison
    initial_speeds = np.sqrt(np.sum(velocities**2, axis=1))
    
    # Store positions for animation
    positions_history = []

    # --- Main Simulation Loop ---
    for step in range(n_steps):
        # Store positions for animation
        if step % 20 == 0:
            positions_history.append(positions.copy())

        # 1. Calculate Doppler force and scattering rate
        doppler_force, scattering_rate = calculate_doppler_force(positions, velocities, LASER_DETUNING, LASER_INTENSITY, k, LASER_BEAM_WAIST)
        
        # 2. Calculate MOT trapping force
        trap_force = -MOT_TRAP_STRENGTH * positions

        # 3. Total force
        total_force = doppler_force + trap_force
        
        # 4. Update velocities from total force (Euler method)
        acceleration = total_force / ATOMIC_MASS
        velocities += acceleration * DT
        
        # 5. Add random recoil from spontaneous emission
        num_scatters = np.random.poisson(scattering_rate * DT)
        total_scatters = np.sum(num_scatters)
        
        if total_scatters > 0:
            # Generate random angles for all scattered photons
            thetas = np.random.uniform(0, 2 * np.pi, total_scatters)
            
            # Calculate recoil momenta
            recoil_momenta_x = HBAR * k * np.cos(thetas)
            recoil_momenta_y = HBAR * k * np.sin(thetas)
            
            # Assign recoils to atoms
            atom_indices = np.repeat(np.arange(N_ATOMS), num_scatters)
            
            # Sum the recoil momenta for each atom
            total_recoil_x = np.bincount(atom_indices, weights=recoil_momenta_x, minlength=N_ATOMS)
            total_recoil_y = np.bincount(atom_indices, weights=recoil_momenta_y, minlength=N_ATOMS)
            
            # Update velocities
            velocities[:, 0] += total_recoil_x / ATOMIC_MASS
            velocities[:, 1] += total_recoil_y / ATOMIC_MASS

        # 6. Update positions (Euler method)
        positions += velocities * DT
        
        # Optional: Add boundary conditions (e.g., periodic or reflective)
        
    # --- Analysis and Visualization ---
    final_speeds = np.sqrt(np.sum(velocities**2, axis=1))

    # --- Print initial and final speed stats ---
    avg_initial_speed = np.mean(initial_speeds)
    std_initial_speed = np.std(initial_speeds)
    avg_final_speed = np.mean(final_speeds)
    std_final_speed = np.std(final_speeds)
    print(f"Average initial speed: {avg_initial_speed:.2f} m/s, std: {std_initial_speed:.2f} m/s")
    print(f"Average final speed:   {avg_final_speed:.2f} m/s, std: {std_final_speed:.2f} m/s")

    # Plot final positions and speed distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(positions[:, 0] * 1e3, positions[:, 1] * 1e3, s=5, label='Rb-87 Atoms')
    plt.title("Final Atom Positions")
    plt.xlabel("X position (mm)")
    plt.ylabel("Y position (mm)")
    plt.axis('equal')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(initial_speeds, bins=15, histtype='step', label=f'Initial (T={initial_temp*1e3:.1f} mK)', color='blue')
    plt.hist(final_speeds, bins=15, histtype='step', label='Final', color='red')
    plt.title("Speed Distribution")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Number of Atoms")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simulation_plot.png')

    # --- Create Animation ---
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_xlim(-BOX_SIZE/2 * 1e3, BOX_SIZE/2 * 1e3)
    ax_anim.set_ylim(-BOX_SIZE/2 * 1e3, BOX_SIZE/2 * 1e3)
    ax_anim.set_xlabel("X position (mm)")
    ax_anim.set_ylabel("Y position (mm)")
    ax_anim.set_title("MOT Animation")
    ax_anim.set_aspect('equal')

    scatter = ax_anim.scatter([], [], s=5, label='Rb-87 Atoms')
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, color='black', fontsize=16, ha='left', va='top', bbox=dict(facecolor='none', edgecolor='none', pad=2))
    ax_anim.legend()

    def update(frame):
        scatter.set_offsets(positions_history[frame] * 1e3)
        current_time = frame * 20 * DT  # Each frame represents 20 simulation steps
        time_text.set_text(f'Time: {current_time*1e3:.2f} ms')
        return scatter, time_text

    def update(frame):
        scatter.set_offsets(positions_history[frame] * 1e3)
        current_time = frame * 20 * DT  # Each frame represents 20 simulation steps
        time_text.set_text(f'Time: {current_time*1e3:.2f} ms')
        return scatter, time_text

    anim = animation.FuncAnimation(fig_anim, update, frames=len(positions_history), blit=True)
    
    # Save the animation
    anim.save('mot_animation.gif', writer='imagemagick', fps=15)
    print("Animation saved as mot_animation.gif")


if __name__ == "__main__":
    main()
