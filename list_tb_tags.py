import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def list_tags(log_path):
    print(f"Tags in {log_path}:")
    event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
    event_acc.Reload()
    tags = event_acc.Tags()
    print("Scalars:", tags.get('scalars', []))

if __name__ == '__main__':
    list_tags(sys.argv[1])
