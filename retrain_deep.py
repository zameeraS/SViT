"""
retrain_deep.py
---------------
Re-trains only SViT-Deep with a fairer optimisation recipe:
  - Linear LR warm-up for the first WARMUP_EPOCHS epochs
  - CosineAnnealingLR for the remaining epochs
  - 60 total epochs (more room to converge)

Results are saved to  results/SViT-Deep-Fair/
so existing results/SViT-Deep/ is untouched for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score,
)
from model import SViT
from dataset import get_data_loaders
from train import train_model
from utils import plot_confusion_matrix


# ─── Helpers (same as train_variants.py) ────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference(model, dataloader, device, n_batches=20):
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            if i == 0:
                _ = model(inputs)   # warm-up
                continue
            t0 = time.perf_counter()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / inputs.size(0))
    avg_lat_ms  = np.mean(latencies) * 1000
    throughput  = 1.0 / np.mean(latencies)
    return avg_lat_ms, throughput


def evaluate_variant(model, dataloaders, device, save_dir):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs  = inputs.to(device)
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

    metrics = dict(
        test_acc=acc, precision=precision, recall=recall, f1=f1,
        specificity=specificity, sensitivity=sensitivity, auc=auc,
        f1_per_cls=f1_per_cls.tolist(), classes=classes,
    )

    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write("SViT-Deep-Fair Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Scheduler      : Linear Warmup ({WARMUP_EPOCHS} ep) → CosineAnnealing\n")
        f.write(f"Total Epochs   : {NUM_EPOCHS}\n\n")
        f.write(f"Test Accuracy  : {acc:.4f}\n")
        f.write(f"Specificity    : {specificity:.4f}\n")
        f.write(f"Sensitivity    : {sensitivity:.4f}\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"F1 Score       : {f1:.4f}\n")
        f.write(f"AUC            : {auc:.4f}\n\n")
        f.write("Per-class F1:\n")
        for cls, score in zip(classes, f1_per_cls):
            f.write(f"  {cls:<12}: {score:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"  Report saved → {report_path}")
    return metrics


# ─── Config (module-level so report can reference them) ──────────────────────

BATCH_SIZE     = 64
LEARNING_RATE  = 0.0001
NUM_EPOCHS     = 60      # more room than the original 40
WARMUP_EPOCHS  = 8       # linear ramp from LR/10 → LR over first 8 epochs
DATA_DIR       = 'DATASET_SPLIT'
SAVE_DIR       = os.path.join('results', 'SViT-Deep-Fair')

DEEP_CONFIG = {'vit_layers': 6, 'vit_heads': 8, 'vit_dim': 512}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scheduler    : Linear warmup ({WARMUP_EPOCHS} ep) → CosineAnnealingLR")
    print(f"Total epochs : {NUM_EPOCHS}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    try:
        dataloaders, datasets = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Class weights
    train_dataset = dataloaders['train'].dataset
    class_counts  = [0] * len(train_dataset.classes)
    for label in train_dataset.labels:
        class_counts[label] += 1
    class_weights = [sum(class_counts) / (len(class_counts) * c) for c in class_counts]
    class_weights  = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model      = SViT(num_classes=4, **DEEP_CONFIG).to(device)
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    num_params = count_parameters(model)
    print(f"Trainable parameters: {num_params:,}\n")

    # ── Loss / Optimizer ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Scheduler: LinearLR warmup → CosineAnnealingLR ───────────────────────
    # LinearLR ramps from start_factor × LR up to LR over WARMUP_EPOCHS steps
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(NUM_EPOCHS - WARMUP_EPOCHS),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"{'='*50}")
    print(f"  Training: SViT-Deep-Fair")
    print(f"{'='*50}")

    t_start = time.time()
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS, device=device, save_dir=SAVE_DIR,
    )
    train_time_s = time.time() - t_start

    best_val_acc  = max(history['val_acc'])
    best_val_loss = min(history['val_loss'])

    # ── Inference speed ───────────────────────────────────────────────────────
    print("Measuring inference speed...")
    lat_ms, throughput = measure_inference(model, dataloaders['val'], device)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("Evaluating on test set...")
    metrics = evaluate_variant(model, dataloaders, device, SAVE_DIR)

    # ── Plot LR curve ─────────────────────────────────────────────────────────
    # Simulate LR schedule for visualisation
    lrs = []
    _opt  = optim.Adam([torch.zeros(1)], lr=LEARNING_RATE)
    _w    = LinearLR(_opt, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    _c    = CosineAnnealingLR(_opt, T_max=(NUM_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    _seq  = SequentialLR(_opt, schedulers=[_w, _c], milestones=[WARMUP_EPOCHS])
    for _ in range(NUM_EPOCHS):
        lrs.append(_opt.param_groups[0]['lr'])
        _seq.step()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, NUM_EPOCHS + 1), lrs)
    plt.title('Learning Rate Schedule (SViT-Deep-Fair)')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.grid(True)
    lr_path = os.path.join(SAVE_DIR, 'lr_schedule.png')
    plt.savefig(lr_path)
    plt.close()
    print(f"LR schedule plot saved → {lr_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  SViT-Deep-Fair — Final Results")
    print(f"{'='*50}")
    print(f"  Params          : {num_params:,}")
    print(f"  Train time      : {train_time_s/60:.1f} min")
    print(f"  Best val acc    : {best_val_acc:.4f}")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"  Test accuracy   : {metrics['test_acc']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  AUC             : {metrics['auc']:.4f}")
    print(f"  Sensitivity     : {metrics['sensitivity']:.4f}")
    print(f"  Specificity     : {metrics['specificity']:.4f}")
    print(f"  Latency         : {lat_ms:.3f} ms/img")
    print(f"  Throughput      : {throughput:.1f} img/s")

    # Save CSV summary
    row = {
        'Variant':          'SViT-Deep-Fair',
        'Scheduler':        f'Warmup({WARMUP_EPOCHS}ep)+Cosine',
        'Epochs':           NUM_EPOCHS,
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
    csv_path = os.path.join(SAVE_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"  Summary CSV     → {csv_path}")


if __name__ == '__main__':
    main()
