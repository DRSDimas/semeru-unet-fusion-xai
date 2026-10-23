"""
train.py

Training loop and execution pipeline for the Gated U-Net framework.
Reference: Subsections 3.8.1, 3.8.2, 3.8.3, and 3.9.2.
"""

import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import csv

from .config import (
    get_config,
    get_all_configs,
    resolve_seeds,
    resolve_strategies,
    resolve_years,
    get_checkpoint_path,
    get_summary_path,
    get_history_path,
)
from .data_loader import create_dataloaders
from .models import get_model, get_model_summary
from .losses import get_loss_function
from .metrics import (
    compute_confusion_matrix,
    compute_metrics_from_confusion,
)


# --- Seed Initialization ---
def set_all_seeds(seed):
    """
    Sets Python, NumPy, and PyTorch seeds for reproducibility.
    """
    # See Subsection 3.8.3 (Training Budget and Model Reproducibility)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Class Weights Calculation ---
def compute_class_weights(dataset, num_classes):
    """
    Computes Inverse-Square-Root class weights from the training set.
    """
    # See Subsection 3.8.1 (Loss Function and Class Weighting)
    print("   Computing class weights from training set...")

    class_counts = np.zeros(num_classes, dtype=np.int64)

    for idx in tqdm(range(len(dataset)), desc="   Counting pixels", leave=False):
        _, label, _, _ = dataset[idx]
        unique, counts = np.unique(label.numpy(), return_counts=True)
        for cls, count in zip(unique, counts):
            if 0 <= cls < num_classes:
                class_counts[cls] += count

    total = class_counts.sum()
    weights = []
    
    # Calculate ISR weights
    # See Equation 3.11
    for count in class_counts:
        if count > 0:
            weights.append(1.0 / np.sqrt(count / total))
        else:
            weights.append(1.0)

    class_names = [f"C{i}" for i in range(num_classes)]
    print(f"   Class counts:  " +
          "  ".join(f"{n}={c:,}" for n, c in zip(class_names, class_counts)))
    print(f"   Class weights: " +
          "  ".join(f"{n}={w:.4f}" for n, w in zip(class_names, weights)))

    return weights


# --- Training and Validation Loops ---
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, num_classes):
    """
    Executes one epoch of model training.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_macro_f1 = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels, clouds, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        conf = compute_confusion_matrix(preds, labels, num_classes=num_classes)
        m = compute_metrics_from_confusion(conf)

        total_loss += loss.item()
        total_acc += m['accuracy']
        total_iou += m['miou']
        total_macro_f1 += m['macro_f1']
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f"{m['accuracy']:.4f}",
            'iou': f"{m['miou']:.4f}",
            'f1': f"{m['macro_f1']:.4f}",
        })

    return (total_loss / num_batches,
            total_acc / num_batches,
            total_iou / num_batches,
            total_macro_f1 / num_batches)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, num_classes, compute_facies=False):
    """
    Evaluates model performance on a validation or test set.
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    total_macro_f1 = 0.0
    num_batches = 0

    all_preds, all_labels, all_clouds, all_facies = [], [], [], []

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    for images, labels, clouds, facies in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        conf = compute_confusion_matrix(preds, labels, num_classes=num_classes)
        m = compute_metrics_from_confusion(conf)

        total_loss += loss.item()
        total_acc += m['accuracy']
        total_iou += m['miou']
        total_macro_f1 += m['macro_f1']
        num_batches += 1

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_clouds.append(clouds.cpu())
        all_facies.append(facies.cpu())

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f"{m['accuracy']:.4f}",
            'iou': f"{m['miou']:.4f}",
            'f1': f"{m['macro_f1']:.4f}",
        })

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_facies = torch.cat(all_facies, dim=0)

    global_confusion = compute_confusion_matrix(
        all_preds, all_labels, num_classes=num_classes
    )
    gm = compute_metrics_from_confusion(global_confusion)

    facies_metrics = {}
    if compute_facies:
        # See Subsection 3.10.4 (Gap Concentration per Facies)
        for facies_code in [1, 2, 3, 4, 5]:
            mask = all_facies == facies_code
            if mask.sum() > 0:
                preds_facies = all_preds[mask]
                labels_facies = all_labels[mask]
                conf_facies = compute_confusion_matrix(
                    preds_facies, labels_facies, num_classes=num_classes
                )
                facies_metrics[facies_code] = compute_metrics_from_confusion(conf_facies)

    return {
        'loss': total_loss / num_batches,
        'acc': total_acc / num_batches,
        'iou': total_iou / num_batches,
        'macro_f1': total_macro_f1 / num_batches,
        'macro_precision': gm['macro_precision'],
        'macro_recall': gm['macro_recall'],
        'global_confusion': global_confusion,
        'facies_metrics': facies_metrics,
    }


# --- Main Training Routine ---
def run_training(config):
    """
    Executes the full training and evaluation pipeline for a single configuration.
    """
    name = config['name']
    seed = config['seed']
    year = config['year']
    num_classes = config['num_classes']
    class_names = config['class_names']

    print("\n" + "=" * 80)
    print(f"STARTING TRAINING: {name}")
    print("=" * 80)
    print(f"Year:        {year}")
    print(f"Strategy:    {config['split_type']}")
    print(f"Seed:        {seed}")
    print(f"Num classes: {num_classes}  ({class_names})")
    print(f"Data root:   {config['data_root']}")
    print(f"Device:      {config['device']}")
    print(f"Epochs:      {config['epochs']} (fixed budget, no early stopping)")
    print("=" * 80)
    print("\nCloud Mask Policy:")
    print("  - S2 bands zeroed where cloudy (data_loader.py)")
    print("  - Loss computed on ALL pixels")
    print("  - Cloud mask NOT used to weight or ignore pixels")
    print("=" * 80)

    # Ensure reproducibility
    # See Subsection 3.8.3
    set_all_seeds(seed)

    # Create output directories
    model_dir = config['model_dir']
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize DataLoaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, data_dict = create_dataloaders(config)

    print(f"   Training batches:   {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches:       {len(test_loader)}")

    # Calculate dynamic ISR class weights
    # See Subsection 3.8.1
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_loader.dataset,
                                          num_classes=num_classes)
    config['class_weights'] = class_weights

    # Initialize Model
    print("\nCreating model...")
    model = get_model(config)
    in_channels = config['s1_channels'] + config['s2_channels']
    get_model_summary(model, (in_channels, config['patch_size'], config['patch_size']))

    # Initialize Loss
    print("\nCreating loss function...")
    loss_fn = get_loss_function(config, config['device'])

    # Initialize Optimizer
    # See Subsection 3.8.2
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    print(f"   Optimizer: Adam (lr={config['learning_rate']})")

    # Initialize Learning Rate Scheduler
    # See Subsection 3.8.2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
        threshold=1e-4, cooldown=5, min_lr=1e-6, verbose=True,
    )
    print(f"   LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6)")

    # Setup Training Loop
    device = config['device']
    epochs = config['epochs']

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [], 'train_iou': [],
        'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_iou': [],
        'val_f1': [], 'lr': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    print("\n" + "=" * 80)
    print("TRAINING LOOP")
    print("=" * 80)

    # Execute training epochs
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc, train_iou, train_f1 = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, num_classes
        )

        val = evaluate(model, val_loader, loss_fn, device, num_classes,
                       compute_facies=False)
        val_loss = val['loss']
        val_acc = val['acc']
        val_iou = val['iou']
        val_f1 = val['macro_f1']

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"   Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_iou'].append(train_iou)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        history['lr'].append(new_lr)

        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"IoU: {train_iou:.4f} | F1: {train_f1:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"IoU: {val_iou:.4f} | F1: {val_f1:.4f}")
        print(f"   LR:         {new_lr:.2e}")

        # Checkpoint selection based on validation loss
        # See Subsection 3.8.3
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_iou': val_iou,
                'val_f1': val_f1,
                'config': config,
            }, get_checkpoint_path(year, config['split_type'], seed))
            print(f"   Saved best model (val_loss: {val_loss:.4f})")

    # Load optimal weights for final evaluation
    model.load_state_dict(best_model_state)

    # Final Test Evaluation
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION ON {year} TEST SET")
    print("=" * 80)

    test = evaluate(model, test_loader, loss_fn, device, num_classes,
                    compute_facies=True)
    test_confusion = test['global_confusion']

    print(f"Test Loss:            {test['loss']:.4f}")
    print(f"Test Accuracy:        {test['acc']:.4f}")
    print(f"Test mIoU:            {test['iou']:.4f}")
    print(f"Test Macro F1:        {test['macro_f1']:.4f}")
    print(f"Test Macro Precision: {test['macro_precision']:.4f}")
    print(f"Test Macro Recall:    {test['macro_recall']:.4f}")
    print(f"Best epoch:           {best_epoch}")
    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    print(f"   Classes: " +
          ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))
    print(test_confusion)

    # Save training history
    with open(get_history_path(year, config['split_type'], seed), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        for i in range(len(history['epoch'])):
            writer.writerow({k: history[k][i] for k in history.keys()})

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test['loss'],
        'test_acc': test['acc'],
        'test_iou': test['iou'],
        'test_macro_f1': test['macro_f1'],
        'test_macro_precision': test['macro_precision'],
        'test_macro_recall': test['macro_recall'],
        'test_confusion': test_confusion,
        'test_facies': test['facies_metrics'],
    }, os.path.join(checkpoint_dir, "final_model.pth"))

    # Save summary report
    with open(get_summary_path(year, config['split_type'], seed), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TRAINING SUMMARY: {name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Year:           {year}\n")
        f.write(f"Strategy:       {config['split_type']}\n")
        f.write(f"Seed:           {seed}\n")
        f.write(f"Num classes:    {num_classes}  ({class_names})\n")
        f.write(f"Data root:      {config['data_root']}\n")
        f.write(f"Class weights:  {config.get('class_weights')}\n\n")
        f.write("Cloud Mask Policy:\n")
        f.write("  - S2 bands zeroed where cloudy (data_loader.py)\n")
        f.write("  - Loss computed on all pixels\n")
        f.write("  - Cloud mask NOT used to weight or ignore pixels\n\n")
        f.write("Training Configuration:\n")
        f.write(f"  Epochs (fixed):  {epochs}\n")
        f.write(f"  Early stopping:  disabled\n")
        f.write(f"  LR scheduler:    ReduceLROnPlateau (factor=0.5, patience=10)\n")
        f.write(f"  Final LR:        {new_lr:.2e}\n\n")
        f.write(f"Best epoch:        {best_epoch}\n")
        f.write(f"Best val loss:     {best_val_loss:.4f}\n\n")
        f.write(f"Test Loss:            {test['loss']:.4f}\n")
        f.write(f"Test Accuracy:        {test['acc']:.4f}\n")
        f.write(f"Test mIoU:            {test['iou']:.4f}\n")
        f.write(f"Test Macro F1:        {test['macro_f1']:.4f}\n")
        f.write(f"Test Macro Precision: {test['macro_precision']:.4f}\n")
        f.write(f"Test Macro Recall:    {test['macro_recall']:.4f}\n\n")
        f.write("Confusion Matrix (Rows=Actual, Cols=Predicted):\n")
        f.write(f"   Classes: " +
                ", ".join(f"{i}={n}" for i, n in enumerate(class_names)) + "\n")
        f.write(str(test_confusion))

        if test['facies_metrics']:
            f.write("\n\n" + "=" * 40 + "\n")
            f.write("PER-FACIES TEST METRICS\n")
            f.write("=" * 40 + "\n")
            facies_names = ["Central", "Proximal", "Medial", "Distal", "Inlier"]
            for code, fname in enumerate(facies_names, 1):
                if code in test['facies_metrics']:
                    fm = test['facies_metrics'][code]
                    f.write(f"\n{fname}:\n")
                    f.write(f"  Accuracy:        {fm['accuracy']:.4f}\n")
                    f.write(f"  mIoU:            {fm['miou']:.4f}\n")
                    f.write(f"  Macro F1:        {fm['macro_f1']:.4f}\n")
                    f.write(f"  Macro Precision: {fm['macro_precision']:.4f}\n")
                    f.write(f"  Macro Recall:    {fm['macro_recall']:.4f}\n")
                    f.write("  Confusion Matrix:\n")
                    f.write(str(fm['confusion']) + "\n")

    print(f"\nModel directory: {model_dir}")

    return {
        'name': name,
        'year': year,
        'strategy': config['split_type'],
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_iou': test['iou'],
        'test_macro_f1': test['macro_f1'],
    }


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Gated U-Net across multiple years, strategies, and seeds."
    )
    parser.add_argument("--year", type=str, default="both",
                        choices=["2021", "2022", "both"])
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["random", "blocking", "all"])
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seed list, or 'all'.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (default: 100).")
    args = parser.parse_args()

    print("=" * 80)
    print("TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    years = resolve_years(args.year)
    strategies = resolve_strategies(args.strategy)
    seeds = resolve_seeds(args.seeds)

    configs = get_all_configs(years=years, strategies=strategies, seeds=seeds)

    if args.epochs is not None:
        for cfg in configs:
            cfg['epochs'] = args.epochs
        print(f"\nEpochs overridden to: {args.epochs}")

    print(f"\nYears:      {years}")
    print(f"Strategies: {strategies}")
    print(f"Seeds:      {seeds}")
    print(f"\nModels to train: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"   {i:2d}. {cfg['name']:<30}  "
              f"year={cfg['year']:<5} "
              f"strategy={cfg['split_type']:<8} "
              f"seed={cfg['seed']:<4} "
              f"classes={cfg['num_classes']}")

    if not args.yes:
        print("\nThis will train the model(s) listed above.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            exit()

    all_results = []
    for cfg in configs:
        result = run_training(cfg)
        all_results.append(result)

    # Final summary output
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} | {'BestEp':<7} | {'ValLoss':<9} | "
          f"{'TestIoU':<9} | {'TestF1':<9}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:<30} | {r['best_epoch']:<7} | "
              f"{r['best_val_loss']:<9.4f} | "
              f"{r['test_iou']:<9.4f} | {r['test_macro_f1']:<9.4f}")

    # Per (year, strategy) summary
    print("\n" + "=" * 80)
    print("MEAN +/- STD PER (YEAR, STRATEGY)")
    print("=" * 80)
    for year in years:
        for strategy in strategies:
            subset = [r for r in all_results
                      if r['year'] == year and r['strategy'] == strategy]
            if not subset:
                continue
            ious = np.array([r['test_iou'] for r in subset])
            f1s = np.array([r['test_macro_f1'] for r in subset])
            print(f"  {year} {strategy:<10}  n={len(subset)}  "
                  f"mIoU = {ious.mean():.4f} +/- {ious.std():.4f}  |  "
                  f"F1 = {f1s.mean():.4f} +/- {f1s.std():.4f}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")