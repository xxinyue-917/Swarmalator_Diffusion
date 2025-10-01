#!/usr/bin/env python
"""
Interactive Swarmalator Simulation
===================================
Adjust the parameters at the top to explore different collective behaviors.
Everything is self-contained in this single script for easy experimentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
import time

# ========================================================================
#                        USER ADJUSTABLE PARAMETERS
# ========================================================================

J = 1.0      # Spatial-phase coupling (-2 to 2):
             #   J > 0: Similar phases attract spatially
             #   J = 0: No phase influence on spatial dynamics
             #   J < 0: Similar phases repel spatially

K = 1.0      # Phase coupling strength (-2 to 2):
             #   K > 0: Phase synchronization
             #   K = 0: No phase coupling
             #   K < 0: Phase anti-synchronization

A = 1.0      # Attraction strength parameter

B = 2.0      # Repulsion strength parameter (matching MATLAB default)

BOT_RAD = 0.01  # Bot radius for collision avoidance

N_AGENTS = 100  # Number of agents (10 to 200)

# Simulation parameters
DT = 0.1           # Time step (matching MATLAB)
ANIMATION_SPEED = 50  # Animation interval in ms

# ========================================================================


class SwarmalatorSystem:
    """
    Swarmalator implementation matching MATLAB version exactly.

    The dynamics follow:
    dx/dt = (1/N) * Σ [(A + J*cos(θj-θi))/dist - B/(d*dist)] * (xj-xi)
    dθ/dt = (K/N) * Σ sin(θj-θi)/dist
    """

    def __init__(self, N=100, J=1.0, K=1.0, A=1.0, B=9.0, bot_rad=0.1, dt=0.1):
        self.N = N
        self.J = J  # Spatial-phase coupling
        self.K = K  # Phase coupling strength
        self.A = A  # Attraction parameter
        self.B = B  # Repulsion parameter
        self.bot_rad = bot_rad  # Bot radius
        self.dt = dt  # Time step

        # Initialize states
        self.reset()

    def reset(self):
        """Reset system with random initialization"""
        # Random positions in [-5, 5] matching MATLAB plotLimit
        plot_limit = 5
        self.positions = np.random.uniform(-plot_limit, plot_limit, (self.N, 2))

        # Initialize phases uniformly (matching MATLAB)
        self.phases = np.linspace(2*np.pi/self.N, 2*np.pi, self.N)

        # History for trails
        self.history = []

    def compute_dynamics(self):
        """
        Compute position and phase updates matching MATLAB implementation exactly.
        Returns dPosX, dPosY, dPhase arrays for all agents.
        """
        dPosX = np.zeros(self.N)
        dPosY = np.zeros(self.N)
        dPhase = np.zeros(self.N)

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    # Calculate distance between two bots
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)

                    # Calculate forces (matching MATLAB exactly)
                    F_attr = (self.A + self.J * np.cos(self.phases[j] - self.phases[i])) / dist
                    F_rep = self.B / (dist*dist)  # Make sure bots never go too close
                    F_total = F_attr - F_rep

                    # Calculate position derivatives
                    dPosX[i] += F_total * dx
                    dPosY[i] += F_total * dy

                    # Calculate phase derivative
                    dPhase[i] += self.K * np.sin(self.phases[j] - self.phases[i]) / dist

        return dPosX, dPosY, dPhase

    def step(self):
        """Single time step matching MATLAB implementation exactly"""
        # Compute dynamics for all agents
        dPosX, dPosY, dPhase = self.compute_dynamics()

        # Update positions (matching MATLAB: dt * sum/N)
        for i in range(self.N):
            self.positions[i, 0] += self.dt * dPosX[i] / self.N
            self.positions[i, 1] += self.dt * dPosY[i] / self.N

            # Update and regularize phase
            self.phases[i] += self.dt * dPhase[i] / self.N
            self.phases[i] = np.mod(self.phases[i], 2*np.pi)

        # Store history (keep last 20 steps for trails)
        self.history.append(self.positions.copy())
        if len(self.history) > 20:
            self.history.pop(0)

    def get_order_parameters(self):
        """Calculate order parameters matching MATLAB"""
        # Phase coherence (Kuramoto order parameter)
        phase_order = np.abs(np.mean(np.exp(1j * self.phases)))

        # Calculate collective radius (matching MATLAB)
        center_of_mass = np.mean(self.positions, axis=0)
        distances = np.zeros(self.N)
        for i in range(self.N):
            dx = self.positions[i, 0] - center_of_mass[0]
            dy = self.positions[i, 1] - center_of_mass[1]
            distances[i] = np.sqrt(dx**2 + dy**2)
        collective_radius = np.max(distances)

        return phase_order, collective_radius


def create_visualization(swarm):
    """Create interactive visualization"""
    # Set up figure with subplots
    fig = plt.figure(figsize=(16, 9))

    # Main spatial plot
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_main.set_xlim(-3.5, 3.5)
    ax_main.set_ylim(-3.5, 3.5)
    ax_main.set_xlabel('X Position', fontsize=10)
    ax_main.set_ylabel('Y Position', fontsize=10)
    ax_main.set_title(f'Swarmalator Dynamics (J={J:.1f}, K={K:.1f})', fontsize=12, fontweight='bold')
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)

    # Phase circle plot
    ax_phase = plt.subplot2grid((3, 3), (0, 2))
    ax_phase.set_xlim(-1.3, 1.3)
    ax_phase.set_ylim(-1.3, 1.3)
    ax_phase.set_xlabel('cos(θ)', fontsize=10)
    ax_phase.set_ylabel('sin(θ)', fontsize=10)
    ax_phase.set_title('Phase Distribution', fontsize=10)
    ax_phase.set_aspect('equal')
    circle = Circle((0, 0), 1, fill=False, edgecolor='gray', linewidth=1)
    ax_phase.add_patch(circle)

    # Phase histogram
    ax_hist = plt.subplot2grid((3, 3), (1, 2))
    ax_hist.set_xlim(0, 2*np.pi)
    ax_hist.set_ylim(0, swarm.N/3)
    ax_hist.set_xlabel('Phase (rad)', fontsize=10)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_title('Phase Histogram', fontsize=10)

    # Order parameters plot
    ax_order = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax_order.set_xlim(0, 100)
    ax_order.set_ylim(0, 1.1)
    ax_order.set_xlabel('Time Steps', fontsize=10)
    ax_order.set_ylabel('Order Parameter', fontsize=10)
    ax_order.set_title('Order Parameters Over Time', fontsize=10)
    ax_order.grid(True, alpha=0.3)

    # Initialize plots
    scatter_main = ax_main.scatter([], [], c=[], cmap='hsv', s=60,
                                   vmin=0, vmax=2*np.pi, edgecolors='black', linewidth=0.5)
    scatter_phase = ax_phase.scatter([], [], c=[], cmap='hsv', s=40,
                                     vmin=0, vmax=2*np.pi, edgecolors='black', linewidth=0.5)

    # Order parameter lines
    line_phase, = ax_order.plot([], [], 'b-', label='Phase coherence', linewidth=2)
    line_radius, = ax_order.plot([], [], 'r-', label='Collective radius', linewidth=2)
    ax_order.legend(loc='upper right')

    # Trail lines for main plot
    trail_lines = []
    for i in range(swarm.N):
        line, = ax_main.plot([], [], 'gray', alpha=0.2, linewidth=0.5)
        trail_lines.append(line)

    # History for order parameters
    order_history = {'phase': [], 'radius': [], 'time': []}

    # Text for current state
    text_state = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                              fontsize=10, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Colorbar
    cbar = plt.colorbar(scatter_main, ax=ax_main, label='Phase (rad)', shrink=0.8)

    def update(frame):
        """Animation update function"""
        # Step simulation
        swarm.step()

        # Update main scatter plot
        scatter_main.set_offsets(swarm.positions)
        scatter_main.set_array(swarm.phases)

        # Update trails
        if len(swarm.history) > 1:
            for i, line in enumerate(trail_lines):
                trail_positions = np.array([h[i] for h in swarm.history])
                line.set_data(trail_positions[:, 0], trail_positions[:, 1])

        # Update phase circle
        x_phase = np.cos(swarm.phases)
        y_phase = np.sin(swarm.phases)
        phase_positions = np.column_stack([x_phase, y_phase])
        scatter_phase.set_offsets(phase_positions)
        scatter_phase.set_array(swarm.phases)

        # Update phase histogram
        ax_hist.clear()
        ax_hist.hist(swarm.phases, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax_hist.set_xlim(0, 2*np.pi)
        ax_hist.set_ylim(0, swarm.N/3)
        ax_hist.set_xlabel('Phase (rad)', fontsize=10)
        ax_hist.set_ylabel('Count', fontsize=10)
        ax_hist.set_title('Phase Histogram', fontsize=10)

        # Calculate and store order parameters
        phase_order, collective_radius = swarm.get_order_parameters()
        order_history['phase'].append(phase_order)
        order_history['radius'].append(collective_radius)
        order_history['time'].append(frame)

        # Keep only recent history
        if len(order_history['time']) > 100:
            for key in order_history:
                order_history[key].pop(0)

        # Update order parameter plots
        line_phase.set_data(order_history['time'], order_history['phase'])
        line_radius.set_data(order_history['time'], order_history['radius'])

        # Update x-axis range and y-axis for radius
        if frame > 100:
            ax_order.set_xlim(frame - 100, frame)

        # Adjust y-axis for radius scale
        if order_history['radius']:
            max_radius = max(order_history['radius'])
            ax_order.set_ylim(0, max(1.1, max_radius * 1.1))

        # Update state text
        state_text = f'Step: {frame}\n'
        state_text += f'Phase coherence: {phase_order:.3f}\n'
        state_text += f'Collective radius: {collective_radius:.3f}'
        text_state.set_text(state_text)

        return scatter_main, scatter_phase, line_phase, line_radius, text_state

    # Add parameter info
    param_text = f'Parameters:\n'
    param_text += f'J = {swarm.J:.1f} (spatial-phase coupling)\n'
    param_text += f'K = {swarm.K:.1f} (phase coupling)\n'
    param_text += f'A = {swarm.A:.1f} (attraction)\n'
    param_text += f'B = {swarm.B:.1f} (repulsion)\n'
    param_text += f'Bot radius = {swarm.bot_rad:.2f}\n'
    param_text += f'N = {swarm.N} agents'

    fig.text(0.02, 0.02, param_text, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    return fig, update


def run_simulation():
    """Run the interactive simulation"""
    print("\n" + "=" * 70)
    print(" SWARMALATOR INTERACTIVE SIMULATION ")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  J = {J:.1f} (spatial-phase coupling)")
    print(f"  K = {K:.1f} (phase coupling)")
    print(f"  A = {A:.1f} (attraction strength)")
    print(f"  B = {B:.1f} (repulsion strength)")
    print(f"  N = {N_AGENTS} agents")
    print("\n" + "-" * 70)

    # Expected behavior based on J and K
    if J > 0 and K > 0:
        print("Expected: Static Sync - Agents synchronize and cluster")
    elif J > 0 and K < 0:
        print("Expected: Splintered Phase Wave - Multiple phase groups")
    elif J < 0 and K > 0:
        print("Expected: Static Phase Wave - Phase gradient in space")
    elif J < 0 and K < 0:
        print("Expected: Active Phase Wave - Dynamic patterns")
    elif J == 0:
        print("Expected: No spatial-phase coupling - Independent dynamics")

    print("-" * 70)
    print("\nSimulation starting...")
    print("Close the window to exit.\n")

    # Create swarmalator system
    swarm = SwarmalatorSystem(N=N_AGENTS, J=J, K=K, A=A, B=B, bot_rad=BOT_RAD, dt=DT)

    # Create visualization
    fig, update = create_visualization(swarm)

    # Create animation
    anim = animation.FuncAnimation(fig, update, interval=ANIMATION_SPEED, blit=False)

    # Show plot
    plt.show()

    print("\nSimulation ended.")


if __name__ == "__main__":
    run_simulation()