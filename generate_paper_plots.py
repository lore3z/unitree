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
    """找到目录下最新的子文件夹并返回其包含的 tfevents 文件"""
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "exported"]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None

def plot_reward_curve(log_paths, labels, title, out_filename):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#d62728']  # 蓝色 (12DoF), 红色 (29DoF)
    
    for log_path, label, color in zip(log_paths, labels, colors):
        print(f"Loading {label} from: {log_path}")
        event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
        event_acc.Reload()
        
        tags = event_acc.Tags()['scalars']
        
        # 匹配常用的reward标签 (例如 'Episode/rew_total' 或 'Train/mean_reward')
        target_tag = None
        for tag in ['Episode/rew_total', 'Train/mean_reward', 'rewards/total']:
            if tag in tags:
                target_tag = tag
                break
                
        if target_tag:
            steps = [s.step for s in event_acc.Scalars(target_tag)]
            vals = [s.value for s in event_acc.Scalars(target_tag)]
            plt.plot(steps, vals, label=label, color=color, linewidth=2, alpha=0.9)
        else:
            print(f"Warning: Could not find reward tag in {label}")

    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Total Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_filename)
    print(f"Success! Saved plot as: {out_filename}")

if __name__ == '__main__':
    # 从你的文件树中获取12dof (g1) 和 29dof (g1_arms) 的日志
    log_dir_12dof = "logs/g1"
    log_dir_29dof = "logs/g1_arms"
    
    path_12dof = get_latest_log(log_dir_12dof)
    path_29dof = get_latest_log(log_dir_29dof)
    
    paths = []
    labels = []
    
    if path_12dof:
        paths.append(path_12dof)
        labels.append("G1 12-DoF (Legs only)")
    if path_29dof:
        paths.append(path_29dof)
        labels.append("G1 29-DoF (Full body)")
        
    if paths:
        plot_reward_curve(paths, labels, "Training Reward Comparison", "paper_reward_comparison.png")
    else:
        print("No tensorboard logs found.")
