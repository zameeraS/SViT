import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from model import SViT
from dataset import get_data_loaders
from train import train_model
from utils import plot_confusion_matrix


# ─── Helpers ────────────────────────────────────────────────────────────────

def count_parameters(model):
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference(model, dataloader, device, n_batches=20):
    """
    Returns avg latency (ms/image) and throughput (images/sec)
    averaged over `n_batches` batches from `dataloader`.
    """
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            # Warm-up on first batch
            if i == 0:
                _ = model(inputs)
                continue
            t0 = time.perf_counter()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / inputs.size(0))   # seconds per image

    avg_lat_ms = np.mean(latencies) * 1000                  # ms/image
    throughput  = 1.0 / np.mean(latencies)                  # images/sec
    return avg_lat_ms, throughput


def evaluate_variant(model, dataloaders, device, save_dir):
    """Full test-set evaluation. Returns a metrics dict and saves a report."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs   = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    cm          = confusion_matrix(all_labels, all_preds)
    acc         = accuracy_score(all_labels, all_preds)
    precision   = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall      = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1          = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_per_cls  = f1_score(all_labels, all_preds, average=None, zero_division=0)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    specificity = float(np.mean(TN / (TN + FP + 1e-8)))
    sensitivity = float(np.mean(TP / (TP + FN + 1e-8)))

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except Exception:
        auc = 0.0

    classes = dataloaders['test'].dataset.classes
    plot_confusion_matrix(cm, classes, save_dir=save_dir)

    metrics = {
        'test_acc':    acc,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'auc':         auc,
        'f1_per_cls':  f1_per_cls.tolist(),
        'classes':     classes,
    }

    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("SViT Variant Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test Accuracy : {acc:.4f}\n")
        f.write(f"Specificity   : {specificity:.4f}\n")
        f.write(f"Sensitivity   : {sensitivity:.4f}\n")
        f.write(f"Precision     : {precision:.4f}\n")
        f.write(f"F1 Score      : {f1:.4f}\n")
        f.write(f"AUC           : {auc:.4f}\n\n")
        f.write("Per-class F1:\n")
        for cls, score in zip(classes, f1_per_cls):
            f.write(f"  {cls:<12}: {score:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"  Report saved → {report_path}")

    return metrics


# ─── Comparison Plots ────────────────────────────────────────────────────────

def plot_comparison(all_val_accs, all_val_losses, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, accs in all_val_accs.items():
        axes[0].plot(range(1, len(accs) + 1), accs, label=name)
    axes[0].set_title('Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    for name, losses in all_val_losses.items():
        axes[1].plot(range(1, len(losses) + 1), losses, label=name)
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('SViT Variant Comparison', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_plot.png')
    plt.savefig(path)
    plt.close()
    print(f"Comparison plot saved → {path}")


def save_summary_csv(summary_rows, save_dir):
    path = os.path.join(save_dir, 'variant_summary.csv')
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV saved → {path}")


def print_summary_table(summary_rows):
    if not summary_rows:
        return
    cols = list(summary_rows[0].keys())
    col_w = max(len(c) for c in cols) + 2
    header = " | ".join(c.ljust(col_w) for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        line = " | ".join(str(row[c]).ljust(col_w) for c in cols)
        print(line)
    print("=" * len(header) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Config ────────────────────────────────────────────────────────────────
    BATCH_SIZE    = 64  # 4090 has 24GB VRAM — 64 keeps GPU well-fed
    LEARNING_RATE = 0.0001
    NUM_EPOCHS    = 40          
    STEP_SIZE     = 20          # LR drops at epoch 20
    DATA_DIR      = 'DATASET_SPLIT'
    RESULTS_DIR   = 'results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Variants ───────────────────────────────────────────────────────────────
    variants = {
        'SViT-Light': {'vit_layers': 1, 'vit_heads': 4, 'vit_dim': 512},
        'SViT-Base':  {'vit_layers': 3, 'vit_heads': 8, 'vit_dim': 512},  # Current best
        'SViT-Deep':  {'vit_layers': 6, 'vit_heads': 8, 'vit_dim': 512},
    }

    # ── Data ───────────────────────────────────────────────────────────────────
    print("Loading data loaders (shared across all variants)...")
    try:
        dataloaders, datasets = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Class weights
    print("Calculating class weights...")
    train_dataset = dataloaders['train'].dataset
    class_counts  = [0] * len(train_dataset.classes)
    for label in train_dataset.labels:
        class_counts[label] += 1
    class_weights = [sum(class_counts) / (len(class_counts) * c) for c in class_counts]
    class_weights  = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Per-variant storage ────────────────────────────────────────────────────
    all_val_accs   = {}
    all_val_losses = {}
    summary_rows   = []

    # ── Train each variant ────────────────────────────────────────────────────
    for name, config in variants.items():
        print(f"\n{'='*50}")
        print(f"  Variant : {name}")
        print(f"  Config  : {config}")
        print(f"{'='*50}")

        save_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        # Model
        model      = SViT(num_classes=4, **config).to(device)
        if hasattr(torch, 'compile'):
            print("  Compiling model with torch.compile()...")
            model = torch.compile(model)
        num_params = count_parameters(model)
        print(f"  Trainable parameters: {num_params:,}")

        # Loss / Optimizer / Scheduler
        criterion   = nn.CrossEntropyLoss(weight=class_weights)
        optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler   = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

        # ── Train ────────────────────────────────────────────────────────────
        t_start = time.time()
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler,
            num_epochs=NUM_EPOCHS, device=device, save_dir=save_dir
        )
        train_time_s = time.time() - t_start

        all_val_accs[name]   = history['val_acc']
        all_val_losses[name] = history['val_loss']

        best_val_acc  = max(history['val_acc'])
        best_val_loss = min(history['val_loss'])

        # ── Inference timing ─────────────────────────────────────────────────
        print("  Measuring inference speed...")
        lat_ms, throughput = measure_inference(model, dataloaders['val'], device)
        print(f"  Avg latency : {lat_ms:.3f} ms/image")
        print(f"  Throughput  : {throughput:.1f} images/sec")

        # ── Test evaluation ──────────────────────────────────────────────────
        print("  Evaluating on test set...")
        metrics = evaluate_variant(model, dataloaders, device, save_dir)

        # ── Summarise ────────────────────────────────────────────────────────
        row = {
            'Variant':          name,
            'Params':           f"{num_params:,}",
            'Train_Time_min':   f"{train_time_s/60:.1f}",
            'Best_Val_Acc':     f"{best_val_acc:.4f}",
            'Best_Val_Loss':    f"{best_val_loss:.4f}",
            'Test_Acc':         f"{metrics['test_acc']:.4f}",
            'F1':               f"{metrics['f1']:.4f}",
            'AUC':              f"{metrics['auc']:.4f}",
            'Sensitivity':      f"{metrics['sensitivity']:.4f}",
            'Specificity':      f"{metrics['specificity']:.4f}",
            'Latency_ms/img':   f"{lat_ms:.3f}",
            'Throughput_img/s': f"{throughput:.1f}",
        }
        summary_rows.append(row)

    # ── Final outputs ─────────────────────────────────────────────────────────
    print("\nAll variants trained. Generating outputs...")

    plot_comparison(all_val_accs, all_val_losses, RESULTS_DIR)
    save_summary_csv(summary_rows, RESULTS_DIR)
    print_summary_table(summary_rows)

    print("Done. All results saved under results/<variant-name>/")


if __name__ == '__main__':
    main()
