import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class OCTDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            split (str): One of 'train', 'test', 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Directory {split_dir} does not exist.")

        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning: Class directory {cls_dir} does not exist. Skipping.")
                continue
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle error appropriately
            # For now, let's just return a black image
            image = Image.new('RGB', (227, 227))

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=8):
    """
    Creates DataLoaders for train, val, and test splits.
    Uses separate augmentation transforms for train vs val/test.
    """
    # SqueezeNet requires 227x227 input
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_map = {'train': train_transform, 'val': val_transform, 'test': val_transform}

    datasets = {
        x: OCTDataset(data_dir, split=x, transform=transform_map[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        for x in ['train', 'val', 'test']
    }

    return dataloaders, datasets
