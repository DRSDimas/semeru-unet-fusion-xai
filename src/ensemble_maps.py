"""
ensemble_maps.py

Deep ensemble generation and uncertainty (agreement) mapping.
Reference: Subsections 3.9.2 and 3.9.3.
"""

import os
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from .config import (
    get_config,
    get_all_configs,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    get_inference_dir,
    get_data_root,
    get_class_names,
    get_class_slugs,
    EXPERIMENTS_ROOT,
    SEEDS,
)
from .metrics import compute_confusion_matrix, compute_metrics_from_confusion


# --- Constants ---
ENSEMBLE_ROOT = os.path.join(EXPERIMENTS_ROOT, "ensemble")
FIGURES_DIR = os.path.join(ENSEMBLE_ROOT, "figures")

# Classification palette (works for both 3-class and binary)
CLASS_COLORS_3 = ['#1f77b4', '#2ca02c', '#d62728']   # BG, PDC, PDC+Lahar
CLASS_COLORS_2 = ['#1f77b4', '#d62728']              # BG, Deposit


def get_class_cmap(year):
    """
    Returns a ListedColormap matching the class count for a given year.
    """
    if str(year) == "2021":
        return ListedColormap(CLASS_COLORS_3)
    return ListedColormap(CLASS_COLORS_2)


def get_class_colors(year):
    return CLASS_COLORS_3 if str(year) == "2021" else CLASS_COLORS_2


# --- Data Loading ---
def load_seed_probabilities(year, strategy, seed):
    """
    Loads probability maps for one (year, strategy, seed) combination.
    """
    # See Subsection 3.9.2 (Ensemble Seed and Agreement Map)
    base = get_inference_dir(year, strategy, seed)
    slugs = get_class_slugs(year)
    paths = {slug: os.path.join(base, f"prob_{slug}.tif") for slug in slugs}

    if not all(os.path.exists(p) for p in paths.values()):
        return None

    out = {}
    meta = None
    for slug, p in paths.items():
        with rasterio.open(p) as src:
            out[slug] = src.read(1).astype(np.float32)
            if meta is None:
                meta = src.meta.copy()
    out['meta'] = meta
    return out


def stack_seed_probabilities(year, strategy, seeds=SEEDS):
    """
    Loads and stacks probability maps across all seeds for a given strategy.
    """
    # See Subsection 3.9.2
    seed_stacks = []
    seeds_used = []
    meta = None
    slugs = get_class_slugs(year)

    for seed in seeds:
        p = load_seed_probabilities(year, strategy, seed)
        if p is None:
            print(f"    [skip] {year}_{strategy}_seed{seed}: missing files")
            continue
        stacked = np.stack([p[slug] for slug in slugs], axis=0)  # Shape: (C, H, W)
        seed_stacks.append(stacked)
        seeds_used.append(seed)
        if meta is None:
            meta = p['meta']

    if not seed_stacks:
        return None, None, slugs, []

    probs_stack = np.stack(seed_stacks, axis=0)  # Shape: (n_seeds, C, H, W)
    return probs_stack, meta, slugs, seeds_used


# --- Ensemble Computation ---
def compute_ensemble(probs_stack):
    """
    Computes ensemble probabilities, classification map, and agreement map.
    """
    # Average probabilities across seeds
    # See Equation 3.14 (Ensemble Probability)
    ensemble_prob = probs_stack.mean(axis=0)
    
    # Determine ensemble class via argmax
    # See Equation 3.15 (Ensemble Classification)
    ensemble_class = np.argmax(ensemble_prob, axis=0).astype(np.uint8)

    # Calculate seed agreement (count of seeds predicting any deposit class)
    # See Equation 3.16 (Agreement Map)
    seed_class = np.argmax(probs_stack, axis=1)  # Shape: (n_seeds, H, W)
    deposit_votes = (seed_class > 0).astype(np.uint8)
    agreement = deposit_votes.sum(axis=0).astype(np.uint8)

    return ensemble_prob, ensemble_class, agreement


def save_geotiff(arr, path, meta, dtype=None):
    profile = meta.copy()
    profile.update(count=1, compress='lzw')
    if dtype is not None:
        profile['dtype'] = dtype
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def save_ensemble_outputs(year, strategy, ensemble_prob, ensemble_class,
                          agreement, meta, slugs):
    """
    Saves ensemble probability, classification, and agreement maps to GeoTIFF.
    """
    # See Subsection 3.9.2
    tag = f"{year}_{strategy}"
    save_geotiff(ensemble_class,
                 os.path.join(ENSEMBLE_ROOT, f"{tag}_ensemble_class.tif"),
                 meta, dtype='uint8')
    for i, slug in enumerate(slugs):
        save_geotiff(ensemble_prob[i].astype(np.float32),
                     os.path.join(ENSEMBLE_ROOT, f"{tag}_ensemble_prob_{slug}.tif"),
                     meta, dtype='float32')
    save_geotiff(agreement,
                 os.path.join(ENSEMBLE_ROOT, f"{tag}_agreement.tif"),
                 meta, dtype='uint8')


# --- Ensemble Metrics ---
def compute_ensemble_metrics(year, strategy, ensemble_class, agreement):
    """
    Evaluates the ensemble class map against ground truth labels.
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    label_path = os.path.join(get_data_root(year), "Label.tif")
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int16) - 1

    num_classes = 3 if str(year) == "2021" else 2
    class_names = get_class_names(year)

    # Native-scheme metrics
    conf = compute_confusion_matrix(ensemble_class, label, num_classes=num_classes)
    m = compute_metrics_from_confusion(conf)

    # Binary-collapsed metrics (deposit vs background) for cross-year comparison
    pred_dep = (ensemble_class > 0).astype(np.uint8)
    label_dep = (label > 0).astype(np.uint8)
    label_dep[label == 254] = 254
    conf_bin = compute_confusion_matrix(pred_dep, label_dep, num_classes=2)
    m_bin = compute_metrics_from_confusion(conf_bin)

    # Agreement statistics
    n_seeds = int(agreement.max()) if agreement.max() > 0 else 1
    deposit_px = (agreement > 0)
    unanimous = (agreement == agreement.max()) & deposit_px

    row = {
        'Year': year,
        'Strategy': strategy,
        'NumClasses': num_classes,
        'LabelScheme': 'Binary' if num_classes == 2 else '3-class',

        # Native
        'Accuracy': m['accuracy'],
        'mIoU': m['miou'],
        'Macro_F1': m['macro_f1'],
        'Macro_Precision': m['macro_precision'],
        'Macro_Recall': m['macro_recall'],

        # Binary-collapsed
        'Bin_Accuracy': m_bin['accuracy'],
        'Bin_mIoU': m_bin['miou'],
        'Bin_Macro_F1': m_bin['macro_f1'],
        'Bin_Background_IoU': m_bin['ious'][0],
        'Bin_Deposit_IoU': m_bin['ious'][1],
        'Bin_Deposit_F1': m_bin['f1'][1],

        # Agreement statistics
        'NSeeds': n_seeds,
        'NPixelsAnyDeposit': int(deposit_px.sum()),
        'NPixelsUnanimousDeposit': int(unanimous.sum()),
        'FractionDepositUnanimous': (
            float(unanimous.sum() / deposit_px.sum())
            if deposit_px.sum() > 0 else 0.0
        ),
        'MeanAgreementOnDeposit': (
            float(agreement[deposit_px].mean())
            if deposit_px.sum() > 0 else 0.0
        ),
    }

    # Native per-class IoU columns
    for i, name in enumerate(class_names):
        safe = name.replace("+", "plus").replace(" ", "_")
        row[f"{safe}_IoU"] = m['ious'][i]

    return row


# --- Visualization Functions ---
def plot_ensemble_class(ensemble_class, agreement, year, strategy,
                        output_path):
    """
    Plots the ensemble classification map alongside the seed agreement map.
    """
    # See Subsection 3.9.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    cmap = get_class_cmap(year)
    colors = get_class_colors(year)
    class_names = get_class_names(year)

    ax = axes[0]
    ax.imshow(ensemble_class, cmap=cmap, vmin=0, vmax=len(class_names) - 1,
              interpolation='nearest')
    ax.set_title(f"Ensemble classification — {year} {strategy}", fontsize=12)
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=colors[i], label=class_names[i])
                      for i in range(len(class_names))]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    ax = axes[1]
    n_seeds = int(agreement.max()) if agreement.max() > 0 else 1
    im = ax.imshow(agreement, cmap='viridis', vmin=0, vmax=max(1, n_seeds),
                   interpolation='nearest')
    ax.set_title(f"Seed agreement on deposit — {year} {strategy}", fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        f"Number of seeds predicting deposit (max {n_seeds})", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_grid(cached, kind, output_path, title):
    """
    Plots a 2x2 grid comparing years (rows) and strategies (columns).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    cells = [('2021', 'random'), ('2021', 'blocking'),
             ('2022', 'random'), ('2022', 'blocking')]

    for ax, (year, strategy) in zip(axes, cells):
        arr = cached.get((year, strategy))
        if arr is None:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            continue
        if kind == 'class':
            cmap = get_class_cmap(year)
            class_names = get_class_names(year)
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=len(class_names) - 1,
                      interpolation='nearest')
        else:
            n_seeds = int(arr.max()) if arr.max() > 0 else 1
            ax.imshow(arr, cmap='viridis', vmin=0, vmax=max(1, n_seeds),
                      interpolation='nearest')
        ax.set_title(f"{year} {strategy}", fontsize=12)
        ax.axis('off')

    # Legend only for class grid, using 2021 3-class palette (representative)
    if kind == 'class':
        legend_patches = [mpatches.Patch(color=CLASS_COLORS_3[i], label=n)
                          for i, n in enumerate(["Background", "PDC",
                                                 "PDC+Lahar"])]
        fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Orchestration ---
def run_ensemble_analysis(years, strategies, seeds=SEEDS):
    """
    Iterates through configurations to generate ensemble maps and metrics.
    """
    # See Subsection 3.9.2 (Ensemble Seed and Agreement Map)
    os.makedirs(ENSEMBLE_ROOT, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metric_rows = []
    class_cache = {}
    agreement_cache = {}

    for year in years:
        for strategy in strategies:
            print(f"\n{'='*70}")
            print(f"  ENSEMBLE: {year} / {strategy}")
            print(f"{'='*70}")

            probs_stack, meta, slugs, seeds_used = stack_seed_probabilities(
                year, strategy, seeds
            )
            if probs_stack is None:
                print(f"  No valid seeds. Skipping.")
                continue

            print(f"  Seeds used: {seeds_used}  (n={len(seeds_used)})")

            ensemble_prob, ensemble_class, agreement = compute_ensemble(probs_stack)

            class_names = get_class_names(year)
            counts = [(ensemble_class == c).sum() for c in range(len(class_names))]
            total = ensemble_class.size
            print(f"  Ensemble class distribution:")
            for c, name in enumerate(class_names):
                print(f"    {name:<12} {int(counts[c]):>10,}  "
                      f"({counts[c]/total*100:5.2f}%)")
            
            if (agreement > 0).any():
                mean_agree = agreement[agreement > 0].mean()
                print(f"  Mean agreement on deposit: {mean_agree:.2f} / {agreement.max()}")
            else:
                print("  No deposit pixels detected.")

            save_ensemble_outputs(year, strategy, ensemble_prob, ensemble_class,
                                  agreement, meta, slugs)

            row = compute_ensemble_metrics(year, strategy, ensemble_class, agreement)
            metric_rows.append(row)

            class_cache[(year, strategy)] = ensemble_class
            agreement_cache[(year, strategy)] = agreement

            plot_ensemble_class(
                ensemble_class, agreement, year, strategy,
                os.path.join(FIGURES_DIR, f"{year}_{strategy}_ensemble.png")
            )

    # Save Metrics CSV
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        path = os.path.join(ENSEMBLE_ROOT, "ensemble_metrics.csv")
        df.to_csv(path, index=False)
        print(f"\nEnsemble metrics saved to: {path}")

    # Generate Grid Figures
    if class_cache:
        plot_grid(class_cache, 'class',
                  os.path.join(FIGURES_DIR, "ensemble_comparison_grid.png"),
                  "Ensemble classification maps")
        plot_grid(agreement_cache, 'agreement',
                  os.path.join(FIGURES_DIR, "agreement_comparison_grid.png"),
                  "Seed agreement on deposit")

    # Console Summary
    if metric_rows:
        print("\n" + "=" * 80)
        print("ENSEMBLE SUMMARY")
        print("=" * 80)
        print(f"{'Year':<6} {'Strategy':<10} {'Native mIoU':<13} "
              f"{'Bin mIoU':<11} {'Bin Dep IoU':<13} {'MeanAgree':<11}")
        print("-" * 80)
        for r in metric_rows:
            print(f"{r['Year']:<6} {r['Strategy']:<10} "
                  f"{r['mIoU']:<13.4f} {r['Bin_mIoU']:<11.4f} "
                  f"{r['Bin_Deposit_IoU']:<13.4f} "
                  f"{r['MeanAgreementOnDeposit']:<11.2f}")

    print(f"\nAll outputs in: {ENSEMBLE_ROOT}")


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deep ensemble and uncertainty maps."
    )
    parser.add_argument("--year", type=str, default="both",
                        choices=["2021", "2022", "both"])
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["random", "blocking", "all"])
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seeds, or 'all'.")
    args = parser.parse_args()

    years = resolve_years(args.year)
    strategies = resolve_strategies(args.strategy)
    seeds = resolve_seeds(args.seeds)

    run_ensemble_analysis(years, strategies, seeds)