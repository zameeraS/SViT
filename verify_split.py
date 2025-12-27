import os
from glob import glob

root = 'DATASET_SPLIT'
classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

for split in ['train', 'val', 'test']:
    print(f"--- {split} ---")
    for cls in classes:
        path = os.path.join(root, split, cls)
        if os.path.exists(path):
            count = len(glob(os.path.join(path, '*.*')))
            print(f"{cls}: {count}")
        else:
            print(f"{cls}: Not found")
