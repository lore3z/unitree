import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_log(log_path, title, out_name):
    print(f"Loading {log_path}...")
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    
    if 'Episode/rew_total' in tags or 'Train/mean_reward' in tags:
        rew_tag = 'Episode/rew_total' if 'Episode/rew_total' in tags else 'Train/mean_reward'
    else:
        rew_tag = tags[0] if len(tags) > 0 else None
        
    if rew_tag:
        w_times, step_nums, vals = zip(*event_acc.Scalars(rew_tag))
        plt.figure(figsize=(10, 5))
        plt.plot(step_nums, vals, label='Reward')
        plt.title(f'{title} - Training Reward')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{out_name}_reward.png')
        plt.close()
        print(f"Saved {out_name}_reward.png")
    
