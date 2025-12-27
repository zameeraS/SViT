import os

def count_classes(root_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split {split} not found.")
            continue
            
        print(f"--- {split.upper()} ---")
        total = 0
        counts = {}
        if os.path.exists(split_dir):
            for cls in os.listdir(split_dir):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    counts[cls] = count
                    total += count
        
        for cls, count in counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{cls}: {count} ({percentage:.2f}%)")
        print(f"Total: {total}\n")

if __name__ == "__main__":
    count_classes('DATASET')
