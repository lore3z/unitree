import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'font.family': 'serif',
})

# 指数移动平均平滑函数 (类似TensorBoard前端的表现)
def smooth_data(scalars, weight=0.95):
    if len(scalars) == 0:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def get_latest_log(log_dir):
    if not os.path.exists(log_dir): return None
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "exported"]
    if not subdirs: return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None

def plot_single(log_paths, labels, tags, title, ylabel, out_filename, weight=0.95):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#d62728']  # Blue, Red
    
    for log_path, label, color in zip(log_paths, labels, colors):
        ea = EventAccumulator(log_path, size_guidance={'scalars': 0})
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        
        target_tag = next((t for t in tags if t in available_tags), None)
        if target_tag:
            steps = [s.step for s in ea.Scalars(target_tag)]
            vals = [s.value for s in ea.Scalars(target_tag)]
            smoothed_vals = smooth_data(vals, weight=weight)
            
            # 绘制极度透明的原始数据 (涂鸦背景) - 可选
            plt.plot(steps, vals, color=color, alpha=0.15, linewidth=1)
            # 绘制平滑后的粗实线 (曲线)
            plt.plot(steps, smoothed_vals, label=label, color=color, linewidth=2.5)

    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved smoothed plot: {out_filename}")

def plot_grid(log_paths, labels, out_filename, weight=0.95):
    ea_12 = EventAccumulator(log_paths[0], size_guidance={'scalars': 0})
    ea_12.Reload()
    ea_29 = EventAccumulator(log_paths[1], size_guidance={'scalars': 0})
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
    eas = [ea_12, ea_29]

    for i, (tag, title) in enumerate(metrics):
        ax = axes[i]
        for idx, ea in enumerate(eas):
            if tag in ea.Tags()['scalars']:
                steps = [s.step for s in ea.Scalars(tag)]
                vals = [s.value for s in ea.Scalars(tag)]
                smoothed_vals = smooth_data(vals, weight=weight)
                
                # 原始透明底色 + 粗线平滑曲线
                ax.plot(steps, vals, color=colors[idx], alpha=0.15, linewidth=1)
                ax.plot(steps, smoothed_vals, label=labels[idx], color=colors[idx], linewidth=2.5)
        
        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.set_xlabel('Training Steps', fontsize=14)
        ax.set_ylabel('Reward value', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax.legend(fontsize=12)
            
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved smoothed grid plot: {out_filename}")

if __name__ == '__main__':
    p12 = get_latest_log("logs/g1")
    p29 = get_latest_log("logs/g1_arms")
    
    if p12 and p29:
        paths = [p12, p29]
        labels = ["G1 12-DoF (Legs only)", "G1 29-DoF (Full body)"]
        
        # 使用 0.95 或 0.98 的超高平滑率，滤除高频抖动，保留趋势
        smooth_weight = 0.95 
        
        plot_single(paths, labels, ['Episode/rew_total', 'Train/mean_reward', 'rewards/total'], "Training Reward Comparison", "Total Reward", "paper_reward_comparison.png", weight=0.98)
        plot_single(paths, labels, ["Loss/value_function"], "Value Function Loss Convergence", "Value Loss", "paper_loss_convergence.png", weight=smooth_weight)
        plot_single(paths, labels, ["Train/mean_episode_length"], "Episode Length Convergence", "Mean Episode Length", "paper_episode_length_convergence.png", weight=smooth_weight)
        
        plot_grid(paths, labels, "paper_subrewards_grid.png", weight=smooth_weight)
    else:
        print("Logs not found.")
