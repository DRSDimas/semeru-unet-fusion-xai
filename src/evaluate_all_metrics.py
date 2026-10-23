"""
evaluate_all_metrics.py

Computes evaluation metrics for full-AOI and test-only scopes.
Reference: Subsections 3.9.2, 3.9.3, and 3.10.4.
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
import rasterio

from .config import (
    get_config,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    get_inference_dir,
    get_data_root,
    get_model_dir,
    get_class_names,
    EXPERIMENTS_ROOT,
    FACIES_NAMES,
)
from .metrics import compute_confusion_matrix, compute_metrics_from_confusion


RESULTS_DIR = os.path.join(EXPERIMENTS_ROOT, "results")
PATCH_SIZE = 160


# --- Data Loading ---
def load_label_and_facies(data_root):
    """
    Loads ground truth label and facies map.
    """
    # See Subsection 3.4.4 (UNOSAT Reference Labels)
    data_root_norm = os.path.normpath(data_root)
    parent_root = os.path.dirname(data_root_norm)

    label_path = os.path.join(data_root_norm, "Label.tif")
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int16) - 1

    facies_path = os.path.join(parent_root, "Facies_Class.tif")
    if not os.path.exists(facies_path):
        raise FileNotFoundError(f"Facies file not found: {facies_path}")
    with rasterio.open(facies_path) as src:
        facies = src.read(1).astype(np.int16)

    valid_vals = set(np.unique(label).tolist()) - {254}
    is_binary = valid_vals.issubset({0, 1})

    return label, facies, is_binary


def collapse_to_binary(pred):
    """
    Collapses multi-class prediction to binary: 0 = Background, 1 = Deposit.
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    return (pred > 0).astype(np.uint8)


def build_test_mask(year, strategy, seed, H, W):
    """
    Builds a boolean (H, W) mask of pixels covered by any test patch.
    Returns None if no test assignment CSV exists.
    """
    # See Subsection 3.10.2 (Train-Test Spatial Proximity)
    model_dir = get_model_dir(year, strategy, seed)
    csv_path = os.path.join(model_dir, "test_patch_assignments.csv")
    if not os.path.exists(csv_path):
        return None

    mask = np.zeros((H, W), dtype=bool)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            y = int(float(row[0]))
            x = int(float(row[1]))
            y1 = min(y + PATCH_SIZE, H)
            x1 = min(x + PATCH_SIZE, W)
            mask[y:y1, x:x1] = True
    return mask


# --- Metric Computation on a Subset ---
def compute_metrics_for_mask(label, pred, mask, num_classes):
    """
    Computes native and binary metrics restricted to a mask.
    If mask is None, uses all valid pixels (full-AOI).
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    if mask is None:
        valid = label != 254
    else:
        valid = (label != 254) & mask

    lab = label[valid]
    prd = pred[valid]

    # Native scheme metrics
    conf = compute_confusion_matrix(prd, lab, num_classes=num_classes)
    m_native = compute_metrics_from_confusion(conf)

    # Binary-collapsed metrics for cross-year comparison
    prd_bin = collapse_to_binary(prd)
    lab_bin = (lab > 0).astype(np.uint8)
    conf_bin = compute_confusion_matrix(prd_bin, lab_bin, num_classes=2)
    m_binary = compute_metrics_from_confusion(conf_bin)

    return m_native, m_binary, conf


def compute_per_facies_metrics(label, pred, facies, mask, num_classes):
    """
    Computes per-facies metrics restricted to a mask.
    Includes the confusion matrix and pixel count for each facies.
    """
    # See Subsection 3.10.4 (Gap Concentration per Facies)
    out = {}
    for code in range(1, 6):
        fname = FACIES_NAMES[code - 1]
        f_mask = facies == code

        if mask is not None:
            f_mask = f_mask & mask

        valid = f_mask & (label != 254)
        if valid.sum() == 0:
            continue

        lab_f = label[valid]
        pred_f = pred[valid]

        conf = compute_confusion_matrix(pred_f, lab_f, num_classes=num_classes)
        metrics = compute_metrics_from_confusion(conf)
        metrics['confusion'] = conf
        metrics['n_pixels'] = int(valid.sum())
        out[fname] = metrics
    return out


# --- Row Builders ---
def make_metric_row(year, strategy, seed, num_classes, is_binary,
                    m_native, m_binary, class_names, scope):
    """
    Builds a per-seed metrics row for the CSV output.
    """
    # See Equation 3.17 (IoU) and Equation 3.19 (Precision/Recall/F1)
    row = {
        "Year": year,
        "Strategy": strategy,
        "Seed": seed,
        "Scope": scope,
        "NumClasses": num_classes,
        "LabelScheme": "Binary" if is_binary else "3-class",

        # Native scheme
        "Accuracy": m_native['accuracy'],
        "mIoU": m_native['miou'],
        "Macro_F1": m_native['macro_f1'],
        "Macro_Precision": m_native['macro_precision'],
        "Macro_Recall": m_native['macro_recall'],

        # Binary-collapsed
        "Bin_Accuracy": m_binary['accuracy'],
        "Bin_mIoU": m_binary['miou'],
        "Bin_Macro_F1": m_binary['macro_f1'],
        "Bin_Background_IoU": m_binary['ious'][0],
        "Bin_Deposit_IoU": m_binary['ious'][1],
        "Bin_Deposit_F1": m_binary['f1'][1],
    }
    for i, name in enumerate(class_names):
        safe = name.replace("+", "plus").replace(" ", "_")
        row[f"{safe}_IoU"] = m_native['ious'][i]
        row[f"{safe}_F1"] = m_native['f1'][i]
    return row


def serialize_confusion(conf):
    """
    Serializes a confusion matrix as a string 'a,b,c;d,e,f;g,h,i'.
    """
    return ";".join(",".join(str(int(v)) for v in row) for row in conf)


def make_facies_rows(year, strategy, seed, facies_metrics, scope):
    """
    Builds per-facies rows with serialized confusion matrices.
    """
    # See Subsection 3.10.4
    rows = []
    for fname, fm in facies_metrics.items():
        rows.append({
            "Year": year,
            "Strategy": strategy,
            "Seed": seed,
            "Scope": scope,
            "Facies": fname,
            "NPixels": fm['n_pixels'],
            "Accuracy": fm['accuracy'],
            "mIoU": fm['miou'],
            "Macro_F1": fm['macro_f1'],
            "Macro_Precision": fm['macro_precision'],
            "Macro_Recall": fm['macro_recall'],
            "ConfusionMatrix": serialize_confusion(fm['confusion']),
        })
    return rows


# --- Single Model Evaluation ---
def evaluate_one(year, strategy, seed):
    """
    Evaluates one model at both full-AOI and test-only scopes.
    Returns: (row_full, facies_full, row_test, facies_test)
    """
    tag = f"{year}_{strategy}_seed{seed}"
    print("\n" + "=" * 80)
    print(f"EVALUATING: {tag}")
    print("=" * 80)

    config = get_config(year, strategy, seed)
    num_classes = config['num_classes']
    class_names = get_class_names(year)

    data_root = get_data_root(year)
    label, facies, is_binary = load_label_and_facies(data_root)
    H, W = label.shape

    pred_dir = get_inference_dir(year, strategy, seed)
    class_map_path = os.path.join(pred_dir, "classification_map.tif")
    if not os.path.exists(class_map_path):
        print(f"   ERROR: Prediction not found: {class_map_path}")
        return None, [], None, []

    with rasterio.open(class_map_path) as src:
        pred = src.read(1).astype(np.int16)

    if pred.shape != label.shape:
        print(f"   ERROR: shape mismatch")
        return None, [], None, []

    # Build test mask (union of all test patches for this model)
    test_mask = build_test_mask(year, strategy, seed, H, W)
    if test_mask is None:
        print(f"   ERROR: no test_patch_assignments.csv found")
        return None, [], None, []
    n_test_px = int((test_mask & (label != 254)).sum())
    n_total_px = int((label != 254).sum())
    print(f"   Test pixels: {n_test_px:,} / {n_total_px:,} "
          f"({n_test_px/n_total_px*100:.1f}%)")

    # Full-AOI evaluation
    # See Subsection 3.9.3
    m_nat_full, m_bin_full, _ = compute_metrics_for_mask(
        label, pred, None, num_classes=num_classes
    )
    print(f"\n   FULL-AOI METRICS ({num_classes} classes)")
    print(f"   Accuracy: {m_nat_full['accuracy']:.4f}  "
          f"mIoU: {m_nat_full['miou']:.4f}  "
          f"Macro F1: {m_nat_full['macro_f1']:.4f}")
    for i, name in enumerate(class_names):
        print(f"     {name:<12} IoU: {m_nat_full['ious'][i]:.4f}")

    # Test-only evaluation
    # See Subsection 3.9.3
    m_nat_test, m_bin_test, _ = compute_metrics_for_mask(
        label, pred, test_mask, num_classes=num_classes
    )
    print(f"\n   TEST-ONLY METRICS ({num_classes} classes)")
    print(f"   Accuracy: {m_nat_test['accuracy']:.4f}  "
          f"mIoU: {m_nat_test['miou']:.4f}  "
          f"Macro F1: {m_nat_test['macro_f1']:.4f}")
    for i, name in enumerate(class_names):
        print(f"     {name:<12} IoU: {m_nat_test['ious'][i]:.4f}")

    # Build rows
    row_full = make_metric_row(year, strategy, seed, num_classes, is_binary,
                               m_nat_full, m_bin_full, class_names, scope="fullAOI")
    row_test = make_metric_row(year, strategy, seed, num_classes, is_binary,
                               m_nat_test, m_bin_test, class_names, scope="testonly")

    # Per-facies metrics
    # See Subsection 3.10.4
    facies_metrics_full = compute_per_facies_metrics(
        label, pred, facies, None, num_classes
    )
    facies_metrics_test = compute_per_facies_metrics(
        label, pred, facies, test_mask, num_classes
    )

    facies_rows_full = make_facies_rows(year, strategy, seed,
                                        facies_metrics_full, "fullAOI")
    facies_rows_test = make_facies_rows(year, strategy, seed,
                                        facies_metrics_test, "testonly")

    return row_full, facies_rows_full, row_test, facies_rows_test


# --- Summary Aggregation ---
def build_summary(rows):
    """
    Computes mean and standard deviation per (Year, Strategy).
    """
    # See Subsection 3.8.3 (Training Budget and Model Reproducibility)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    id_cols = {"Year", "Strategy", "Seed", "Scope", "NumClasses", "LabelScheme"}
    metric_cols = [c for c in df.columns if c not in id_cols]

    summary_rows = []
    for (year, strategy), grp in df.groupby(["Year", "Strategy"]):
        row = {
            "Year": year,
            "Strategy": strategy,
            "N": len(grp),
            "NumClasses": int(grp["NumClasses"].iloc[0]),
            "LabelScheme": grp["LabelScheme"].iloc[0],
        }
        for col in metric_cols:
            vals = grp[col].astype(float).values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


# --- Main Orchestrator ---
def run_evaluation(years, strategies, seeds, output_dir=None):
    """
    Iterates through configurations to generate evaluation CSVs and summaries.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("METRICS EVALUATION (full-AOI + test-only)")
    print("=" * 80)
    print(f"Years:      {years}")
    print(f"Strategies: {strategies}")
    print(f"Seeds:      {seeds}")
    print(f"Total runs: {len(years) * len(strategies) * len(seeds)}")

    rows_full, rows_test = [], []
    facies_full, facies_test = [], []

    for year in years:
        for strategy in strategies:
            for seed in seeds:
                rf, ff, rt, ft = evaluate_one(year, strategy, seed)
                if rf is not None:
                    rows_full.append(rf)
                    facies_full.extend(ff)
                    rows_test.append(rt)
                    facies_test.extend(ft)

    if not rows_full:
        print("\nNo successful evaluations.")
        return

    # Write CSVs
    df_rows_full = pd.DataFrame(rows_full)
    df_rows_test = pd.DataFrame(rows_test)
    df_fac_full = pd.DataFrame(facies_full)
    df_fac_test = pd.DataFrame(facies_test)

    df_rows_full.to_csv(os.path.join(output_dir, "per_seed_metrics.csv"), index=False)
    df_rows_test.to_csv(os.path.join(output_dir, "per_seed_metrics_testonly.csv"), index=False)
    df_fac_full.to_csv(os.path.join(output_dir, "facies_metrics.csv"), index=False)
    df_fac_test.to_csv(os.path.join(output_dir, "facies_metrics_testonly.csv"), index=False)

    df_sum_full = build_summary(rows_full)
    df_sum_test = build_summary(rows_test)
    df_sum_full.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    df_sum_test.to_csv(os.path.join(output_dir, "summary_metrics_testonly.csv"), index=False)

    print(f"\nSaved 6 CSVs to: {output_dir}")
    print(f"  per_seed_metrics.csv              (full-AOI)")
    print(f"  per_seed_metrics_testonly.csv     (test-only)")
    print(f"  summary_metrics.csv               (full-AOI)")
    print(f"  summary_metrics_testonly.csv      (test-only)")
    print(f"  facies_metrics.csv                (full-AOI)")
    print(f"  facies_metrics_testonly.csv       (test-only)")

    # Print Summary
    print("\n" + "=" * 80)
    print("FULL-AOI SUMMARY (mean +/- std across seeds)")
    print("=" * 80)
    print(f"{'Year':<6} {'Strategy':<10} {'N':<3} "
          f"{'Native mIoU':<18} {'Bin mIoU':<18} {'Dep IoU':<18}")
    print("-" * 90)
    for _, row in df_sum_full.iterrows():
        nat = f"{row['mIoU_mean']:.4f} +/- {row['mIoU_std']:.4f}"
        bin_ = f"{row['Bin_mIoU_mean']:.4f} +/- {row['Bin_mIoU_std']:.4f}"
        dep = f"{row['Bin_Deposit_IoU_mean']:.4f} +/- {row['Bin_Deposit_IoU_std']:.4f}"
        print(f"{row['Year']:<6} {row['Strategy']:<10} {int(row['N']):<3} "
              f"{nat:<18} {bin_:<18} {dep:<18}")

    print("\n" + "=" * 80)
    print("TEST-ONLY SUMMARY (mean +/- std across seeds)")
    print("=" * 80)
    print(f"{'Year':<6} {'Strategy':<10} {'N':<3} "
          f"{'Native mIoU':<18} {'Bin mIoU':<18} {'Dep IoU':<18}")
    print("-" * 90)
    for _, row in df_sum_test.iterrows():
        nat = f"{row['mIoU_mean']:.4f} +/- {row['mIoU_std']:.4f}"
        bin_ = f"{row['Bin_mIoU_mean']:.4f} +/- {row['Bin_mIoU_std']:.4f}"
        dep = f"{row['Bin_Deposit_IoU_mean']:.4f} +/- {row['Bin_Deposit_IoU_std']:.4f}"
        print(f"{row['Year']:<6} {row['Strategy']:<10} {int(row['N']):<3} "
              f"{nat:<18} {bin_:<18} {dep:<18}")

    # Test-only per-facies coverage
    print("\nTest-only per-facies coverage (which facies have test patches per model):")
    for year in years:
        for strategy in strategies:
            sub = df_fac_test[(df_fac_test['Year'] == year) &
                              (df_fac_test['Strategy'] == strategy)]
            if sub.empty:
                continue
            facies_present = sorted(sub['Facies'].unique())
            print(f"  {year} {strategy:<10} -> {facies_present}")

    # Native-class breakdown for 2021
    print("\nNative-class IoU by (year, strategy) - deposit classes only:")
    print("  (test-only numbers)")
    for _, row in df_sum_test.iterrows():
        y = row['Year']
        if row['NumClasses'] == 3:
            pdc = row.get('PDC_IoU_mean', float('nan'))
            pl = row.get('PDCplusLahar_IoU_mean', float('nan'))
            print(f"  {y} {row['Strategy']:<10}  "
                  f"PDC IoU = {pdc:.4f}  |  PDC+Lahar IoU = {pl:.4f}")
        else:
            dep = row.get('Deposit_IoU_mean', float('nan'))
            print(f"  {y} {row['Strategy']:<10}  "
                  f"Deposit IoU = {dep:.4f}")

    print("\nAll evaluations complete.")


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate models (full-AOI + test-only)."
    )
    parser.add_argument("--year", type=str, default="both",
                        choices=["2021", "2022", "both"])
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["random", "blocking", "all"])
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seeds, or 'all'.")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    years = resolve_years(args.year)
    strategies = resolve_strategies(args.strategy)
    seeds = resolve_seeds(args.seeds)

    run_evaluation(years, strategies, seeds, output_dir=args.output_dir)