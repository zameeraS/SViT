import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='results'):
    """
    Plots training and validation loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def plot_data_distribution(datasets, save_dir='results'):
    """
    Plots the distribution of classes in train, val, and test sets.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    classes = datasets['train'].classes
    
    counts = {split: [0] * len(classes) for split in splits}
    
    for split in splits:
        labels = datasets[split].labels
        for label in labels:
            counts[split][label] += 1
            
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, counts['train'], width, label='Train')
    plt.bar(x, counts['val'], width, label='Val')
    plt.bar(x + width, counts['test'], width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Data Distribution across Splits')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'data_distribution.png'))
    plt.close()

def plot_confusion_matrix(cm, classes, save_dir='results'):
    """
    Plots the confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
