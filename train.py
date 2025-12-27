import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from dataset import get_data_loaders
from model import SViT
from utils import plot_training_curves, plot_confusion_matrix, plot_data_distribution

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') # Stopping criterion based on validation loss

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    best_val_acc = 0.0

    try:
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_accs.append(epoch_acc.item())
                else:
                    val_losses.append(epoch_loss)
                    val_accs.append(epoch_acc.item())
                    
                    # Checkpoint if accuracy improved
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch}_acc_{epoch_acc:.4f}.pth')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss,
                            'acc': epoch_acc
                        }, checkpoint_path)
                        print(f"  Validation accuracy improved to {epoch_acc:.4f}. Saved checkpoint to {checkpoint_path}")
            
            # Update plots at the end of each epoch
            plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving plots and best model...")
        torch.save(model.state_dict(), 'interrupted_model.pth')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save plots
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model

def evaluate_model(model, dataloaders, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Specificity, Sensitivity, Precision, F1
    # For multiclass, we can calculate these per class or macro-averaged
    # The prompt implies single values, so likely macro-average or weighted
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro') # Sensitivity
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Specificity is TN / (TN + FP). In multiclass, it's complex. 
    # We can calculate it per class and average.
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    specificity = np.mean(TN / (TN + FP))
    sensitivity = np.mean(TP / (TP + FN)) # Should match recall

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except:
        auc = 0.0
        print("Could not calculate AUC (maybe only one class in test set?)")

    print(f'Test Accuracy: {acc:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    
    # Plot Confusion Matrix
    classes = dataloaders['test'].dataset.classes
    plot_confusion_matrix(cm, classes)

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    DATA_DIR = 'DATASET_SPLIT'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    try:
        dataloaders, datasets = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
        # Plot data distribution
        plot_data_distribution(datasets)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'DATASET' folder exists with train/val/test subfolders.")
        return

    # Model
    model = SViT(num_classes=4)
    model = model.to(device)

    # Calculate Class Weights
    print("Calculating class weights...")
    train_dataset = dataloaders['train'].dataset
    class_counts = [0] * len(train_dataset.classes)
    for label in train_dataset.labels:
        class_counts[label] += 1
    
    class_weights = [sum(class_counts) / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class Weights: {class_weights}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Schedule: Step decay
    # Assuming step size of 30 based on typical 100 epoch runs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Train
    model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS, device=device)

    # Save the final model (or best model returned by train_model)
    torch.save(model.state_dict(), 'final_model.pth')
    print("Saved final model to final_model.pth")

    # Evaluate
    evaluate_model(model, dataloaders, device=device)

if __name__ == '__main__':
    main()
