import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 设置论文风格的绘图参数
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'font.family': 'serif',
})

def get_latest_log(log_dir):
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "exported"]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None

def plot_metric(log_paths, labels, tags_to_check, title, ylabel, out_filename):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#d62728']  # 蓝色 (12DoF), 红色 (29DoF)
    
    for log_path, label, color in zip(log_paths, labels, colors):
        event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        
        target_tag = None
        for tag in tags_to_check:
            if tag in tags:
                target_tag = tag
                break
                
        if target_tag:
            steps = [s.step for s in event_acc.Scalars(target_tag)]
            vals = [s.value for s in event_acc.Scalars(target_tag)]
            plt.plot(steps, vals, label=label, color=color, linewidth=2, alpha=0.9)
        else:
            print(f"Warning: Could not find any of tags {tags_to_check} in {label}")

    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_filename)
    print(f"Success! Saved convergence plot as: {out_filename}")

if __name__ == '__main__':
    log_dir_12dof = "logs/g1"
    log_dir_29dof = "logs/g1_arms"
    
    path_12dof = get_latest_log(log_dir_12dof)
    path_29dof = get_latest_log(log_dir_29dof)
    
    paths, labels = [], []
    if path_12dof:
        paths.append(path_12dof)
        labels.append("G1 12-DoF (Legs only)")
    if path_29dof:
        paths.append(path_29dof)
        labels.append("G1 29-DoF (Full body)")
        
    if paths:
        # Plot Loss Value Function (Convergence)
        plot_metric(paths, labels, ["Loss/value_function"], "Value Function Loss Convergence", "Value Loss", "paper_loss_convergence.png")
        
        # Plot Episode Length (Survival time)
        plot_metric(paths, labels, ["Train/mean_episode_length"], "Episode Length Convergence", "Mean Episode Length", "paper_episode_length_convergence.png")
        
        # Plot Total Reward (Mean Reward)
        plot_metric(paths, labels, ["Train/mean_reward"], "Reward Convergence Curve", "Mean Reward", "paper_reward_convergence.png")
    else:
        print("No tensorboard logs found.")
