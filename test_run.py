import torch
import torch.nn as nn
import torch.optim as optim
from model import SViT
from torch.utils.data import DataLoader, TensorDataset
import os

def test_run():
    print("Starting test run with dummy data...")
    
    # Create dummy data
    # 4 classes, 227x227 images
    train_x = torch.randn(10, 3, 227, 227)
    train_y = torch.randint(0, 4, (10,))
    val_x = torch.randn(5, 3, 227, 227)
    val_y = torch.randint(0, 4, (5,))
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=2),
        'val': DataLoader(val_dataset, batch_size=2),
        'test': DataLoader(val_dataset, batch_size=2) # Use val as test for dummy
    }
    
    # Mock dataset classes for utils
    dataloaders['test'].dataset.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SViT(num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print("Running 1 epoch of training...")
    model.train()
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print("Training step successful.")
    
    print("Running evaluation...")
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
    print("Evaluation step successful.")
    print("Test run complete. The code structure is valid.")

if __name__ == '__main__':
    test_run()
