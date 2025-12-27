import os
import shutil
import random

def rebalance_dataset(root_dir, val_split=0.1):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Train or Val directory not found.")
        return

    print(f"Moving {val_split*100}% of training data to validation...")
    
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cls in classes:
        src_cls_dir = os.path.join(train_dir, cls)
        dst_cls_dir = os.path.join(val_dir, cls)
        
        if not os.path.exists(dst_cls_dir):
            os.makedirs(dst_cls_dir)
            
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_to_move = int(len(images) * val_split)
        
        images_to_move = random.sample(images, num_to_move)
        
        print(f"Class {cls}: Moving {num_to_move} images.")
        
        for img in images_to_move:
            src_path = os.path.join(src_cls_dir, img)
            dst_path = os.path.join(dst_cls_dir, img)
            shutil.move(src_path, dst_path)
            
    print("Rebalancing complete.")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    rebalance_dataset('DATASET')
