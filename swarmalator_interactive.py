#!/usr/bin/env python
"""
Interactive Swarmalator Simulation with Real-Time Control
=========================================================
Adjust all parameters including animation speed using sliders.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# Initial parameters
N_AGENTS = 100
DT = 0.1


class SwarmalatorSystem:
    """Swarmalator implementation matching MATLAB version"""

    def __init__(self, N=100):
        self.N = N
        self.J = 1.0
        self.K = 1.0
        self.A = 1.0
        self.B = 2.0
        self.dt = DT
        self.reset()

    def reset(self):
        """Reset with random positions and uniform phases"""
        self.positions = np.random.uniform(-5, 5, (self.N, 2))
        self.phases = np.linspace(2*np.pi/self.N, 2*np.pi, self.N)

    def step(self):
        """Single time step matching MATLAB implementation"""
        dPosX = np.zeros(self.N)
        dPosY = np.zeros(self.N)
        dPhase = np.zeros(self.N)

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist > 0.001:
                        F_attr = (self.A + self.J * np.cos(self.phases[j] - self.phases[i])) / dist
                        F_rep = self.B / (dist * dist)
                        F_total = F_attr - F_rep

                        dPosX[i] += F_total * dx
                        dPosY[i] += F_total * dy
                        dPhase[i] += self.K * np.sin(self.phases[j] - self.phases[i]) / dist

        # Update positions and phases
        self.positions[:, 0] += self.dt * dPosX / self.N
        self.positions[:, 1] += self.dt * dPosY / self.N
        self.phases += self.dt * dPhase / self.N
        self.phases = np.mod(self.phases, 2*np.pi)

    def get_metrics(self):
        """Calculate order parameters"""
        phase_order = np.abs(np.mean(np.exp(1j * self.phases)))
        center_of_mass = np.mean(self.positions, axis=0)
        distances = np.linalg.norm(self.positions - center_of_mass, axis=1)
        collective_radius = np.max(distances)
        return phase_order, collective_radius


# Create main figure
fig = plt.figure(figsize=(14, 9))
plt.subplots_adjust(bottom=0.35, hspace=0.3)

# Main spatial plot
ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
ax_main.set_xlim(-6, 6)
ax_main.set_ylim(-6, 6)
ax_main.set_aspect('equal')
ax_main.set_xlabel('X Position')
ax_main.set_ylabel('Y Position')
ax_main.set_title('Swarmalator Dynamics', fontweight='bold', fontsize=14)
ax_main.grid(True, alpha=0.3)

# Phase distribution plot
ax_phase = plt.subplot2grid((2, 3), (0, 2))
ax_phase.set_xlim(-1.2, 1.2)
ax_phase.set_ylim(-1.2, 1.2)
ax_phase.set_aspect('equal')
ax_phase.set_xlabel('cos(θ)')
ax_phase.set_ylabel('sin(θ)')
ax_phase.set_title('Phase Distribution')
circle = Circle((0, 0), 1, fill=False, edgecolor='gray')
ax_phase.add_patch(circle)

# Phase histogram
ax_hist = plt.subplot2grid((2, 3), (1, 2))
ax_hist.set_xlim(0, 2*np.pi)
ax_hist.set_xlabel('Phase (rad)')
ax_hist.set_ylabel('Count')
ax_hist.set_title('Phase Histogram')

# Initialize swarmalator
swarm = SwarmalatorSystem(N=N_AGENTS)

# Create scatter plots
scatter_main = ax_main.scatter(swarm.positions[:, 0], swarm.positions[:, 1],
                              c=swarm.phases, cmap='hsv', s=40,
                              vmin=0, vmax=2*np.pi, edgecolors='black', linewidth=0.3)

scatter_phase = ax_phase.scatter(np.cos(swarm.phases), np.sin(swarm.phases),
                                c=swarm.phases, cmap='hsv', s=30,
                                vmin=0, vmax=2*np.pi, edgecolors='black', linewidth=0.3)

plt.colorbar(scatter_main, ax=ax_main, label='Phase (rad)', shrink=0.8)

# Info text
info_text = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Create sliders
slider_ax_j = plt.axes([0.15, 0.25, 0.7, 0.025])
slider_ax_k = plt.axes([0.15, 0.21, 0.7, 0.025])
slider_ax_a = plt.axes([0.15, 0.17, 0.7, 0.025])
slider_ax_b = plt.axes([0.15, 0.13, 0.7, 0.025])
slider_ax_speed = plt.axes([0.15, 0.07, 0.7, 0.025])

slider_j = Slider(slider_ax_j, 'J (phase-space)', -2.0, 2.0, valinit=1.0, valstep=0.1, color='purple')
slider_k = Slider(slider_ax_k, 'K (phase sync)', -2.0, 2.0, valinit=1.0, valstep=0.1, color='blue')
slider_a = Slider(slider_ax_a, 'A (attraction)', 0.0, 3.0, valinit=1.0, valstep=0.1, color='green')
slider_b = Slider(slider_ax_b, 'B (repulsion)', 0.0, 10.0, valinit=2.0, valstep=0.5, color='red')
slider_speed = Slider(slider_ax_speed, 'Frame Speed (FPS)', 5, 60, valinit=20, valstep=5, color='orange')

# Buttons
btn_ax_pause = plt.axes([0.35, 0.02, 0.1, 0.03])
btn_ax_reset = plt.axes([0.55, 0.02, 0.1, 0.03])
btn_pause = Button(btn_ax_pause, 'Pause', color='lightblue')
btn_reset = Button(btn_ax_reset, 'Reset', color='lightgray')

# Animation control
animation_running = [True]
animation_obj = [None]

def update_params(val=None):
    """Update swarm parameters from sliders"""
    swarm.J = slider_j.val
    swarm.K = slider_k.val
    swarm.A = slider_a.val
    swarm.B = slider_b.val

def update_speed(val=None):
    """Update frame rate (how often display updates)"""
    if animation_obj[0] is not None:
        # Convert FPS to interval in milliseconds
        # FPS = frames per second, interval = milliseconds per frame
        fps = slider_speed.val
        interval_ms = 1000.0 / fps  # Convert FPS to milliseconds
        animation_obj[0].event_source.interval = interval_ms

def toggle_pause(event):
    """Toggle animation pause/play"""
    animation_running[0] = not animation_running[0]
    btn_pause.label.set_text('Play' if not animation_running[0] else 'Pause')
    btn_pause.color = 'lightgreen' if not animation_running[0] else 'lightblue'
    fig.canvas.draw_idle()

def reset(event):
    """Reset simulation and sliders"""
    swarm.reset()
    slider_j.reset()
    slider_k.reset()
    slider_a.reset()
    slider_b.reset()
    slider_speed.reset()

# Connect controls
slider_j.on_changed(update_params)
slider_k.on_changed(update_params)
slider_a.on_changed(update_params)
slider_b.on_changed(update_params)
slider_speed.on_changed(update_speed)
btn_pause.on_clicked(toggle_pause)
btn_reset.on_clicked(reset)

def animate(frame):
    """Animation update function"""
    if animation_running[0]:
        # Step simulation
        swarm.step()

        # Update main scatter
        scatter_main.set_offsets(swarm.positions)
        scatter_main.set_array(swarm.phases)

        # Update phase scatter
        x_phase = np.cos(swarm.phases)
        y_phase = np.sin(swarm.phases)
        scatter_phase.set_offsets(np.column_stack([x_phase, y_phase]))
        scatter_phase.set_array(swarm.phases)

        # Update histogram
        ax_hist.clear()
        ax_hist.hist(swarm.phases, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax_hist.set_xlim(0, 2*np.pi)
        ax_hist.set_ylim(0, N_AGENTS/3)
        ax_hist.set_xlabel('Phase (rad)')
        ax_hist.set_ylabel('Count')

        # Update info text
        phase_order, radius = swarm.get_metrics()
        info = f'J={swarm.J:.1f}, K={swarm.K:.1f}, A={swarm.A:.1f}, B={swarm.B:.1f}\n'
        info += f'Phase coherence: {phase_order:.3f}\n'
        info += f'Collective radius: {radius:.2f}'

        if swarm.J > 0 and swarm.K > 0:
            info += '\n→ Sync tendency'
        elif swarm.J < 0 and swarm.K > 0:
            info += '\n→ Phase wave'
        elif swarm.J == 0:
            info += '\n→ No phase coupling'

        info_text.set_text(info)

    return scatter_main, scatter_phase, info_text

# Create animation
anim = FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
animation_obj[0] = anim

print("\n" + "="*70)
print("INTERACTIVE SWARMALATOR SIMULATION")
print("="*70)
print("\n🎛️  CONTROLS:")
print("  • Drag sliders to change parameters in real-time")
print("  • Use 'Animation Speed' slider to control simulation speed")
print("  • Click 'Pause' to freeze, 'Play' to resume")
print("  • Click 'Reset' to restart with new random positions")
print("\nPARAMETER GUIDE:")
print("  J > 0: Similar phases attract spatially")
print("  J < 0: Similar phases repel spatially")
print("  K > 0: Phases synchronize")
print("  K < 0: Phases anti-synchronize")
print("="*70)

plt.show()