import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'font.family': 'serif',
})

def get_latest_log(log_dir):
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "exported"]
    if not subdirs: return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None

def plot_subrewards_grid(log_path_12, log_path_29, out_filename):
    print("Loading data for grid plot...")
    ea_12 = EventAccumulator(log_path_12, size_guidance={'scalars': 0})
    ea_12.Reload()
    ea_29 = EventAccumulator(log_path_29, size_guidance={'scalars': 0})
    ea_29.Reload()

    metrics = [
        ('Episode/rew_tracking_lin_vel', 'Linear Velocity Tracking'),
        ('Episode/rew_tracking_ang_vel', 'Angular Velocity Tracking'),
        ('Episode/rew_torques', 'Torque Penalty (Energy)'),
        ('Episode/rew_action_rate', 'Action Rate (Smoothness)'),
        ('Episode/rew_orientation', 'Posture Penalty (Orientation)'),
        ('Episode/rew_base_height', 'Base Height Maintenance')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#d62728']
    labels = ["G1 12-DoF", "G1 29-DoF"]
    eas = [ea_12, ea_29]

    for i, (tag, title) in enumerate(metrics):
        ax = axes[i]
        for idx, ea in enumerate(eas):
            if tag in ea.Tags()['scalars']:
                steps = [s.step for s in ea.Scalars(tag)]
                vals = [s.value for s in ea.Scalars(tag)]
                ax.plot(steps, vals, label=labels[idx], color=colors[idx], linewidth=1.5, alpha=0.85)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Reward value')
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:  # Only put legend on the first plot to save space
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(out_filename)
    print(f"Success! Grid plot saved as: {out_filename}")

if __name__ == '__main__':
    p12 = get_latest_log("logs/g1")
    p29 = get_latest_log("logs/g1_arms")
    if p12 and p29:
        plot_subrewards_grid(p12, p29, "paper_subrewards_grid.png")
    else:
        print("Logs not found.")
