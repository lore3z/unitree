import sys
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_log(log_dir):
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d != "exported"]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    events = glob.glob(os.path.join(latest_subdir, 'events.out.tfevents.*'))
    return events[0] if events else None

p12 = get_latest_log("logs/g1")
p29 = get_latest_log("logs/g1_arms")

for name, path in [("12-DoF", p12), ("29-DoF", p29)]:
    ea = EventAccumulator(path)
    ea.Reload()
    if 'Loss/value_function' in ea.Tags()['scalars']:
        s = ea.Scalars('Loss/value_function')
        print(f"{name} max step: {s[-1].step}")
