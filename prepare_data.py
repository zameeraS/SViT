import os
import shutil
import random
from glob import glob

def prepare_data(source_dir, target_dir):
    """
    Reorganizes the dataset into a new directory structure.
    - Test: 250 images per class
    - Train: 70% of the remaining images
    - Val: 30% of the remaining images
    """
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    if os.path.exists(target_dir):
        print(f"Target directory {target_dir} already exists. Please remove it or choose a different name.")
        return

    os.makedirs(target_dir)
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(target_dir, split, cls))

    print(f"Scanning {source_dir}...")

    for cls in classes:
        # Collect all images for this class from all existing splits
        all_images = []
        for split in ['train', 'val', 'test']:
            cls_dir = os.path.join(source_dir, split, cls)
            if os.path.exists(cls_dir):
                images = glob(os.path.join(cls_dir, '*.*'))
                # Filter for valid image extensions just in case
                images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                all_images.extend(images)
        
        print(f"Class {cls}: Found {len(all_images)} total images.")
        
        if len(all_images) < 250:
            print(f"Warning: Not enough images in {cls} to satisfy 250 test images requirement.")
            # Handle this case? For now, let's just use what we have for test and warn.
            # But strictly following instructions, we should probably fail or adjust.
            # Let's assume there are enough images based on the context.
        
        random.shuffle(all_images)
        
        # Split
        test_images = all_images[:250]
        remaining = all_images[250:]
        
        num_train = int(len(remaining) * 0.7)
        train_images = remaining[:num_train]
        val_images = remaining[num_train:]
        
        print(f"  - Test: {len(test_images)}")
        print(f"  - Train: {len(train_images)}")
        print(f"  - Val: {len(val_images)}")
        
        # Copy
        for img_path in test_images:
            shutil.copy(img_path, os.path.join(target_dir, 'test', cls, os.path.basename(img_path)))
            
        for img_path in train_images:
            shutil.copy(img_path, os.path.join(target_dir, 'train', cls, os.path.basename(img_path)))
            
        for img_path in val_images:
            shutil.copy(img_path, os.path.join(target_dir, 'val', cls, os.path.basename(img_path)))

    print(f"Data preparation complete. New dataset at {target_dir}")

if __name__ == "__main__":
    random.seed(42) # Ensure reproducibility
    # Assuming the current dataset is in 'DATASET' and we want to create 'DATASET_SPLIT'
    # We need to check where the user's dataset actually is. 
    # Based on previous file views, it seems to be 'DATASET'.
    
    SOURCE_DATASET = 'DATASET'
    TARGET_DATASET = 'DATASET_SPLIT'
    
    if not os.path.exists(SOURCE_DATASET):
         print(f"Error: Source dataset '{SOURCE_DATASET}' not found.")
    else:
        prepare_data(SOURCE_DATASET, TARGET_DATASET)
