import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
from model import SViT
from dataset import get_data_loaders
from train import train_model

def main():
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20 # Reduced for comparative study
    DATA_DIR = 'DATASET_SPLIT'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Variants to test
    variants = {
        'SViT-Light': {'vit_layers': 1, 'vit_heads': 4, 'vit_dim': 512},
        'SViT-Base':  {'vit_layers': 3, 'vit_heads': 8, 'vit_dim': 512}, # Current Best
        'SViT-Deep':  {'vit_layers': 6, 'vit_heads': 8, 'vit_dim': 512}
    }

    # Data Loaders (Shared across variants)
    try:
        dataloaders, datasets = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate Class Weights (Shared)
    print("Calculating class weights...")
    train_dataset = dataloaders['train'].dataset
    class_counts = [0] * len(train_dataset.classes)
    for label in train_dataset.labels:
        class_counts[label] += 1
    
    class_weights = [sum(class_counts) / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class Weights: {class_weights}")

    # Store results for comparison
    all_val_accs = {}

    for name, config in variants.items():
        print(f"\n{'='*20}")
        print(f"Training Variant: {name}")
        print(f"Config: {config}")
        print(f"{'='*20}")

        # Initialize Model
        model = SViT(num_classes=4, **config)
        model = model.to(device)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Scheduler
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Adjusted for 20 epochs

        # Save Directory
        save_dir = os.path.join('results', name)
        os.makedirs(save_dir, exist_ok=True)

        # Train
        model, history = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS, device=device, save_dir=save_dir)
        
        # Store validation accuracy
        all_val_accs[name] = history['val_acc']

    print("\nAll variants trained. Generating comparison plot...")
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    for name, accs in all_val_accs.items():
        epochs = range(1, len(accs) + 1)
        plt.plot(epochs, accs, label=name)
    
    plt.title('Validation Accuracy Comparison (ViT Variants)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', 'comparison_plot.png'))
    plt.close()
    print("Comparison plot saved to results/comparison_plot.png")

if __name__ == '__main__':
    main()
