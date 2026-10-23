"""
leakage_analysis.py

Computes structural diagnostics to separate spatial leakage from coverage problems.
Reference: Subsections 3.10.1, 3.10.2, 3.10.3, 3.10.4, and 3.10.5.
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from .config import (
    get_model_dir,
    get_inference_dir,
    get_data_root,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    EXPERIMENTS_ROOT,
    FACIES_NAMES,
)


# --- Constants ---
LEAKAGE_ROOT = os.path.join(EXPERIMENTS_ROOT, "leakage")
FIGURES_DIR = os.path.join(LEAKAGE_ROOT, "figures")
RESULTS_DIR = os.path.join(EXPERIMENTS_ROOT, "results")

PATCH_SIZE = 160
STRIDE = 80

ROOK_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --- I/O Helpers ---
def load_label(year):
    """
    Loads the ground truth label raster.
    """
    # See Subsection 3.4.4 (UNOSAT Reference Labels)
    path = os.path.join(get_data_root(year), "Label.tif")
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16) - 1


def load_image_stack(year):
    """
    Loads and concatenates the Sentinel-1 and Sentinel-2 normalized rasters.
    """
    # See Subsection 3.4.5 (Collocation and Normalization)
    root = get_data_root(year)
    with rasterio.open(os.path.join(root, "S1_Backscatter_Norm.tif")) as src:
        s1 = src.read().astype(np.float32)
    with rasterio.open(os.path.join(root, "S2_Reflectance_Norm.tif")) as src:
        s2 = src.read().astype(np.float32)
    stack = np.concatenate([s1, s2], axis=0)
    return np.transpose(stack, (1, 2, 0))


def read_patch_assignments(year, strategy, seed):
    """
    Reads the CSV files containing patch assignments for each split.
    """
    # See Subsection 3.10.2 (Train-Test Spatial Proximity)
    model_dir = get_model_dir(year, strategy, seed)
    out = {'train': [], 'val': [], 'test': []}
    for split in ['train', 'val', 'test']:
        path = os.path.join(model_dir, f"{split}_patch_assignments.csv")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                out[split].append((int(float(row[0])), int(float(row[1]))))
    return out


def build_patch_grid(H, W, patch_size, stride):
    """
    Generates the patch grid coordinates for the given raster dimensions.
    """
    # See Subsection 3.6.1 (Patch Extraction)
    ys = list(range(0, H - patch_size + 1, stride))
    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    xs = list(range(0, W - patch_size + 1, stride))
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)
    return [(y, x) for y in ys for x in xs]


def patch_majority_label(label, y, x, patch_size):
    """
    Determines the majority label within a single patch.
    """
    # See Subsection 3.10.1 (Label Autocorrelation)
    patch = label[y:y+patch_size, x:x+patch_size]
    valid = patch[patch != 254]
    if valid.size == 0:
        return -1
    u, c = np.unique(valid, return_counts=True)
    return int(u[np.argmax(c)])


def morans_i_rook(residual):
    """
    Calculates Moran's I using a rook neighborhood.
    """
    # See Subsection 3.10.1
    # See Equation 3.21
    valid = ~np.isnan(residual)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return float('nan'), n_valid

    mean = np.nanmean(residual[valid])
    z = np.where(valid, residual - mean, 0.0)

    num, W_total = 0.0, 0.0
    for dy, dx in ROOK_OFFSETS:
        y0, y1 = max(0, -dy), min(z.shape[0], z.shape[0] - dy)
        x0, x1 = max(0, -dx), min(z.shape[1], z.shape[1] - dx)
        a = z[y0:y1, x0:x1]
        b = z[y0+dy:y1+dy, x0+dx:x1+dx]
        m = valid[y0:y1, x0:x1] & valid[y0+dy:y1+dy, x0+dx:x1+dx]
        num += float((a * b)[m].sum())
        W_total += float(m.sum())

    denom = float((z[valid] ** 2).sum())
    if denom == 0 or W_total == 0:
        return float('nan'), n_valid
    return float((n_valid / W_total) * (num / denom)), n_valid


# --- Structural Diagnostics ---
def compute_link1(year):
    """
    Computes label autocorrelation metrics (adjacent fraction and Moran's I).
    """
    # See Subsection 3.10.1 (Label Autocorrelation)
    print(f"\n  [Diagnostic 1] Label autocorrelation - {year}")
    label = load_label(year)
    H, W = label.shape
    grid = build_patch_grid(H, W, PATCH_SIZE, STRIDE)

    labels = {p: patch_majority_label(label, p[0], p[1], PATCH_SIZE)
              for p in grid}

    grid_set = set(grid)
    matching, total = 0, 0
    for (y, x) in grid:
        la = labels[(y, x)]
        if la < 0:
            continue
        for dy, dx in ROOK_OFFSETS:
            neighbor = (y + dy * STRIDE, x + dx * STRIDE)
            if neighbor in grid_set:
                lb = labels[neighbor]
                if lb < 0:
                    continue
                total += 1
                if la == lb:
                    matching += 1

    fraction_adj = matching / total if total > 0 else float('nan')
    print(f"    Adjacent patch pairs sharing label: "
          f"{matching}/{total} = {fraction_adj:.4f}")

    label_f = label.astype(np.float32).copy()
    label_f[label_f == 254] = np.nan
    moran_label, n_valid = morans_i_rook(label_f)
    print(f"    Moran's I on label map: {moran_label:.4f}  "
          f"({n_valid:,} valid pixels)")

    return {
        'Year': year,
        'FractionAdjacentSharingLabel': fraction_adj,
        'MoransI_LabelMap': moran_label,
        'NValidPixels': n_valid,
        'NPatchPairsChecked': total,
    }


def compute_link2(year, strategy, seed):
    """
    Computes the distribution of distances from test patches to the nearest training patch.
    """
    # See Subsection 3.10.2 (Train-Test Spatial Proximity)
    assignments = read_patch_assignments(year, strategy, seed)
    train = assignments.get('train', [])
    test = assignments.get('test', [])
    if not train or not test:
        return None, None

    train_centers = np.array([(y + PATCH_SIZE / 2, x + PATCH_SIZE / 2)
                              for y, x in train])
    test_centers = np.array([(y + PATCH_SIZE / 2, x + PATCH_SIZE / 2)
                             for y, x in test])

    tree = cKDTree(train_centers)
    distances, _ = tree.query(test_centers)

    dist_df = pd.DataFrame({
        'Year': year, 'Strategy': strategy, 'Seed': seed,
        'y': [t[0] for t in test], 'x': [t[1] for t in test],
        'DistancePx': distances,
        'DistanceKm': distances * 10.0 / 1000.0,
    })

    stats_row = {
        'Year': year, 'Strategy': strategy, 'Seed': seed,
        'NTest': len(test), 'NTrain': len(train),
        'MinDistPx': float(distances.min()),
        'MaxDistPx': float(distances.max()),
        'MeanDistPx': float(distances.mean()),
        'MedianDistPx': float(np.median(distances)),
        'P25DistPx': float(np.percentile(distances, 25)),
        'P75DistPx': float(np.percentile(distances, 75)),
        'FracWithin80Px': float((distances <= 80).mean()),
        'FracWithin160Px': float((distances <= 160).mean()),
        'FracWithin240Px': float((distances <= 240).mean()),
    }
    return stats_row, dist_df


def compute_link3(year, n_sample=400, seed=42):
    """
    Computes spatial smoothness of predictor features and its correlation with label disagreement.
    """
    # See Subsection 3.10.3 (Predictor Feature Smoothness)
    print(f"\n  [Diagnostic 3] Feature smoothness - {year}")
    label = load_label(year)
    image = load_image_stack(year)
    H, W = label.shape
    grid = build_patch_grid(H, W, PATCH_SIZE, STRIDE)

    feats, labs = {}, {}
    for (y, x) in grid:
        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
        feats[(y, x)] = patch.reshape(-1, patch.shape[2]).mean(axis=0)
        labs[(y, x)] = patch_majority_label(label, y, x, PATCH_SIZE)

    rng = np.random.RandomState(seed)
    n_total = len(grid)
    n_pairs = min(n_sample * (n_sample - 1) // 2, 20000)
    idx_a = rng.randint(0, n_total, size=n_pairs)
    idx_b = rng.randint(0, n_total, size=n_pairs)
    keep = idx_a != idx_b
    idx_a, idx_b = idx_a[keep], idx_b[keep]

    spatial_d, feature_d, label_disagree = [], [], []
    for ia, ib in zip(idx_a, idx_b):
        pa, pb = grid[ia], grid[ib]
        sd = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
        fd = float(np.linalg.norm(feats[pa] - feats[pb]))
        la, lb = labs[pa], labs[pb]
        if la < 0 or lb < 0:
            continue
        spatial_d.append(sd)
        feature_d.append(fd)
        label_disagree.append(0.0 if la == lb else 1.0)

    spatial_d = np.array(spatial_d)
    feature_d = np.array(feature_d)
    label_disagree = np.array(label_disagree)

    if len(spatial_d) < 100:
        print(f"    Too few valid pairs ({len(spatial_d)}). Skipping.")
        return None

    # Pearson correlations
    # See Equation 3.22
    r_sf, _ = pearsonr(spatial_d, feature_d)
    r_sl, _ = pearsonr(spatial_d, label_disagree)
    r_fl, _ = pearsonr(feature_d, label_disagree)

    print(f"    N pairs:                       {len(spatial_d):,}")
    print(f"    r(spatial, feature):           {r_sf:+.4f}")
    print(f"    r(spatial, label_disagree):    {r_sl:+.4f}")
    print(f"    r(feature, label_disagree):    {r_fl:+.4f}")

    return {
        'Year': year,
        'NPairs': len(spatial_d),
        'R_Spatial_Feature': float(r_sf),
        'R_Spatial_LabelDisagree': float(r_sl),
        'R_Feature_LabelDisagree': float(r_fl),
        'MeanSpatialDist': float(spatial_d.mean()),
        'MeanFeatureDist': float(feature_d.mean()),
        'FractionAdjacentLabelDisagree': (
            float(label_disagree[spatial_d <= STRIDE].mean())
            if (spatial_d <= STRIDE).sum() > 0 else float('nan')
        ),
    }


def compute_link4(years):
    """
    Computes the mIoU gap between random and blocking splits per facies.
    """
    # See Subsection 3.10.4 (Gap Concentration per Facies)
    # See Equation 3.23
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4 - Test-only per-facies gap")
    print("=" * 80)

    path = os.path.join(RESULTS_DIR, "facies_metrics_testonly.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        print(f"  Run evaluate_all_metrics.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for year in years:
        sub = df[df['Year'].astype(str) == str(year)]
        if sub.empty:
            continue

        facies_present = {}
        for strategy in ['random', 'blocking']:
            mask = sub['Strategy'] == strategy
            facies_present[strategy] = set(sub[mask]['Facies'].unique())

        common = sorted(facies_present['random'] & facies_present['blocking'])
        print(f"\n  {year} facies with test-only coverage in both strategies: "
              f"{common}")
        print(f"    Random coverage:   {sorted(facies_present['random'])}")
        print(f"    Blocking coverage: {sorted(facies_present['blocking'])}")

        for facies in common:
            r_rows = sub[(sub['Strategy'] == 'random') &
                         (sub['Facies'] == facies)]
            b_rows = sub[(sub['Strategy'] == 'blocking') &
                         (sub['Facies'] == facies)]
            if len(r_rows) == 0 or len(b_rows) == 0:
                continue

            r_mean = float(r_rows['mIoU'].mean())
            r_std = float(r_rows['mIoU'].std(ddof=0))
            b_mean = float(b_rows['mIoU'].mean())
            b_std = float(b_rows['mIoU'].std(ddof=0))
            r_npix = float(r_rows['NPixels'].mean())
            b_npix = float(b_rows['NPixels'].mean())

            print(f"      {facies:<10} "
                  f"random={r_mean:.4f}+/-{r_std:.4f} (n_px~{r_npix:.0f})  "
                  f"blocking={b_mean:.4f}+/-{b_std:.4f} (n_px~{b_npix:.0f})  "
                  f"gap={r_mean - b_mean:+.4f}")

            rows.append({
                'Year': year,
                'Facies': facies,
                'Random_mIoU': r_mean,
                'Random_Std': r_std,
                'Random_NPixels': r_npix,
                'Blocking_mIoU': b_mean,
                'Blocking_Std': b_std,
                'Blocking_NPixels': b_npix,
                'Gap': r_mean - b_mean,
                'NSeeds': min(len(r_rows), len(b_rows)),
            })

    return pd.DataFrame(rows)


def plot_link4_gap_concentration(df_link4, output_path):
    """
    Visualizes the mIoU gap per facies between random and blocking strategies.
    """
    if df_link4.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    years = sorted(df_link4['Year'].astype(str).unique())
    facies_list = sorted(df_link4['Facies'].unique())
    width = 0.35
    x = np.arange(len(facies_list))

    for i, year in enumerate(years):
        sub = df_link4[df_link4['Year'].astype(str) == year]
        gaps = []
        for f in facies_list:
            row = sub[sub['Facies'] == f]
            gaps.append(row['Gap'].values[0] if len(row) > 0 else 0.0)
        ax.bar(x + i * width - width/2, gaps, width, label=str(year))

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(facies_list)
    ax.set_ylabel("mIoU gap (random - blocking)")
    ax.set_title("Test-only per-facies gap")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Discussion Outputs ---
def compute_q2_fasies_fullAOI(years):
    """
    Aggregates full-AOI per-facies mIoU and Macro F1 across seeds.
    """
    # See Subsection 3.10.5 (Evidence Convergence Interpretation Framework)
    print("\n" + "=" * 80)
    print("Full-AOI per-facies mIoU")
    print("=" * 80)

    path = os.path.join(RESULTS_DIR, "facies_metrics.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []
    for year in years:
        sub = df[df['Year'].astype(str) == str(year)]
        if sub.empty:
            continue
        for strategy in ['random', 'blocking']:
            s = sub[sub['Strategy'] == strategy]
            for facies in sorted(s['Facies'].unique()):
                f = s[s['Facies'] == facies]
                rows.append({
                    'Year': year,
                    'Strategy': strategy,
                    'Facies': facies,
                    'mIoU_mean': float(f['mIoU'].mean()),
                    'mIoU_std': float(f['mIoU'].std(ddof=0)),
                    'Macro_F1_mean': float(f['Macro_F1'].mean()),
                    'NSeeds': len(f),
                })

    df_out = pd.DataFrame(rows)
    print(f"  Aggregated {len(df_out)} (year, strategy, facies) rows")
    return df_out


def compute_q2_fasies_overprediction(years):
    """
    Computes per-facies overprediction metrics (BG->deposit FP rate and precision).
    """
    # See Subsection 3.10.5
    print("\n" + "=" * 80)
    print("Per-facies overprediction metrics")
    print("=" * 80)

    path = os.path.join(RESULTS_DIR, "facies_metrics.csv")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    rows = []

    for _, r in df.iterrows():
        conf_str = r.get('ConfusionMatrix', '')
        if not isinstance(conf_str, str) or not conf_str:
            continue
        try:
            matrix = np.array([list(map(int, row.split(',')))
                               for row in conf_str.split(';')])
        except Exception:
            continue

        n_classes = matrix.shape[0]
        if n_classes not in (2, 3):
            continue

        total_bg_actual = int(matrix[0, :].sum())
        bg_to_deposit = int(matrix[0, 1:].sum())

        predicted_deposit = int(matrix[:, 1:].sum())
        actual_deposit_from_pred = int(matrix[1:, 1:].sum())
        precision_dep = (actual_deposit_from_pred / predicted_deposit
                         if predicted_deposit > 0 else float('nan'))

        fp_rate = (bg_to_deposit / total_bg_actual
                   if total_bg_actual > 0 else float('nan'))

        rows.append({
            'Year': r['Year'],
            'Strategy': r['Strategy'],
            'Seed': int(r['Seed']),
            'Facies': r['Facies'],
            'NPixels': int(r['NPixels']),
            'ActualBG': total_bg_actual,
            'BG_to_deposit': bg_to_deposit,
            'FPRate': fp_rate,
            'DepositPrecision': precision_dep,
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        agg = (df_out.groupby(['Year', 'Strategy', 'Facies'])
                     .agg(NSeeds=('Seed', 'count'),
                          FPRate_mean=('FPRate', 'mean'),
                          FPRate_std=('FPRate', 'std'),
                          Precision_mean=('DepositPrecision', 'mean'),
                          Precision_std=('DepositPrecision', 'std'))
                     .reset_index())
        print("\n  Per-facies overprediction summary:")
        print(agg.to_string(index=False))
    return df_out


def compute_q2_gap_shrinkage(years):
    """
    Computes the mIoU gap across scopes (full-AOI vs test-only) and metrics.
    """
    # See Subsection 3.10.5
    print("\n" + "=" * 80)
    print("Paired-year gap (random - blocking)")
    print("=" * 80)

    rows = []
    for scope, fname in [("fullAOI", "summary_metrics.csv"),
                        ("testonly", "summary_metrics_testonly.csv")]:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        for year in years:
            y = str(year)
            r = df[(df['Year'].astype(str) == y) & (df['Strategy'] == 'random')]
            b = df[(df['Year'].astype(str) == y) & (df['Strategy'] == 'blocking')]
            if r.empty or b.empty:
                continue

            rows.append({
                'Scope': scope,
                'Year': year,
                'Random_native_mIoU': float(r['mIoU_mean'].values[0]),
                'Blocking_native_mIoU': float(b['mIoU_mean'].values[0]),
                'Gap_native': float(r['mIoU_mean'].values[0] - b['mIoU_mean'].values[0]),
                'Gap_binary': float(r['Bin_mIoU_mean'].values[0] - b['Bin_mIoU_mean'].values[0]),
                'Gap_deposit_IoU': float(r['Bin_Deposit_IoU_mean'].values[0] - b['Bin_Deposit_IoU_mean'].values[0]),
            })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print("\n  Gap random-blocking per (scope, year):")
        print(df_out.to_string(index=False))
    return df_out


# --- Figures ---
def plot_proximity_histograms(prox_df, output_path):
    """
    Plots histograms of patch distances to the nearest training patch.
    """
    if prox_df.empty:
        return
    combos = prox_df[['Year', 'Strategy']].drop_duplicates().values.tolist()
    n = len(combos)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for i, (year, strategy) in enumerate(combos):
        ax = axes[i // cols][i % cols]
        sub = prox_df[(prox_df['Year'] == year) &
                      (prox_df['Strategy'] == strategy)]
        ax.hist(sub['DistancePx'], bins=30, alpha=0.7,
                color='#1f77b4' if strategy == 'random' else '#ff7f0e')
        ax.set_title(f"{year} - {strategy}")
        ax.set_xlabel("Distance to nearest training patch (px)")
        ax.set_ylabel("N test patches")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fasies_overprediction(df, output_path):
    """
    Plots the per-facies false positive rate from background to deposit.
    """
    if df.empty:
        return

    years = sorted(df['Year'].astype(str).unique())
    facies_list = sorted(df['Facies'].unique())
    n_years = len(years)

    fig, axes = plt.subplots(1, n_years, figsize=(7 * n_years, 6), squeeze=False)
    axes = axes.flatten()

    for ax, year in zip(axes, years):
        sub = df[df['Year'].astype(str) == year]
        width = 0.35
        x = np.arange(len(facies_list))

        r_data, b_data = [], []
        for f in facies_list:
            r_vals = sub[(sub['Strategy'] == 'random') &
                         (sub['Facies'] == f)]['FPRate'].values
            b_vals = sub[(sub['Strategy'] == 'blocking') &
                         (sub['Facies'] == f)]['FPRate'].values
            r_data.append(float(r_vals.mean()) if len(r_vals) else 0.0)
            b_data.append(float(b_vals.mean()) if len(b_vals) else 0.0)

        ax.bar(x - width/2, r_data, width, label='random', color='#1f77b4')
        ax.bar(x + width/2, b_data, width, label='blocking', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels(facies_list, rotation=30, ha='right')
        ax.set_ylabel("BG -> deposit false positive rate")
        ax.set_title(f"{year}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle("Per-facies overprediction (full-AOI)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Main Orchestrator ---
def run_analysis(years, strategies, seeds):
    """
    Executes the full structural diagnostics pipeline.
    """
    # See Subsection 3.10.5 (Evidence Convergence Interpretation Framework)
    os.makedirs(LEAKAGE_ROOT, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Label Autocorrelation
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1 - Label autocorrelation")
    print("=" * 80)
    df_link1 = pd.DataFrame([compute_link1(y) for y in years])
    df_link1.to_csv(os.path.join(LEAKAGE_ROOT,
                                 "link1_label_autocorrelation.csv"),
                    index=False)
    print(f"\nSaved: link1_label_autocorrelation.csv")

    # 2. Train-Test Proximity
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2 - Train-test proximity")
    print("=" * 80)
    link2_stats, link2_dists = [], []
    for year in years:
        for strategy in strategies:
            for seed in seeds:
                print(f"\n  [Diagnostic 2] {year}_{strategy}_seed{seed}")
                s, d = compute_link2(year, strategy, seed)
                if s is not None:
                    link2_stats.append(s)
                    link2_dists.append(d)
                    print(f"    n_test={s['NTest']}  "
                          f"median={s['MedianDistPx']:.0f}px  "
                          f"frac<=160px={s['FracWithin160Px']:.3f}  "
                          f"frac<=80px={s['FracWithin80Px']:.3f}")
    df_link2_stats = pd.DataFrame(link2_stats)
    df_link2_stats.to_csv(os.path.join(LEAKAGE_ROOT,
                                       "link2_proximity_stats.csv"),
                          index=False)
    print(f"\nSaved: link2_proximity_stats.csv")

    if link2_dists:
        df_link2_dists = pd.concat(link2_dists, ignore_index=True)
        df_link2_dists.to_csv(
            os.path.join(LEAKAGE_ROOT, "link2_proximity_distribution.csv"),
            index=False)
        print(f"Saved: link2_proximity_distribution.csv")
        plot_proximity_histograms(
            df_link2_dists,
            os.path.join(FIGURES_DIR, "link2_proximity_histograms.png"))

    # 3. Feature Smoothness
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3 - Feature smoothness")
    print("=" * 80)
    link3_rows = [compute_link3(y) for y in years]
    df_link3 = pd.DataFrame([r for r in link3_rows if r is not None])
    df_link3.to_csv(os.path.join(LEAKAGE_ROOT, "link3_feature_smoothness.csv"),
                    index=False)
    print(f"\nSaved: link3_feature_smoothness.csv")

    # 4. Gap Concentration
    df_link4 = compute_link4(years)
    df_link4.to_csv(os.path.join(LEAKAGE_ROOT, "link4_gap_concentration.csv"),
                    index=False)
    print(f"\nSaved: link4_gap_concentration.csv")
    if not df_link4.empty:
        plot_link4_gap_concentration(
            df_link4,
            os.path.join(FIGURES_DIR, "link4_gap_concentration.png"))

    # Discussion Outputs
    print("\n" + "=" * 80)
    print("DISCUSSION OUTPUTS")
    print("=" * 80)

    df_fasies_full = compute_q2_fasies_fullAOI(years)
    df_fasies_full.to_csv(
        os.path.join(LEAKAGE_ROOT, "q2_discussion_fasies_fullAOI.csv"),
        index=False)
    print(f"Saved: q2_discussion_fasies_fullAOI.csv")

    df_overpred = compute_q2_fasies_overprediction(years)
    df_overpred.to_csv(
        os.path.join(LEAKAGE_ROOT, "q2_discussion_fasies_overprediction.csv"),
        index=False)
    print(f"Saved: q2_discussion_fasies_overprediction.csv")
    if not df_overpred.empty:
        plot_fasies_overprediction(
            df_overpred,
            os.path.join(FIGURES_DIR, "q2_fasies_overprediction.png"))

    df_gap = compute_q2_gap_shrinkage(years)
    df_gap.to_csv(
        os.path.join(LEAKAGE_ROOT, "q2_discussion_gap_shrinkage.csv"),
        index=False)
    print(f"Saved: q2_discussion_gap_shrinkage.csv")

    # Summary CSV
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summary_rows = []
    for year in years:
        y = str(year)

        r1 = df_link1[df_link1['Year'].astype(str) == y]
        if not r1.empty:
            summary_rows.append({
                'Year': year, 'Source': '1_label_autocorr',
                'Metric': 'FractionAdjacentSharingLabel',
                'Mean': float(r1['FractionAdjacentSharingLabel'].iloc[0]),
                'Std': 0.0, 'N': 1,
            })

        for strategy in strategies:
            sub = df_link2_stats[
                (df_link2_stats['Year'].astype(str) == y) &
                (df_link2_stats['Strategy'] == strategy)
            ]
            if not sub.empty:
                summary_rows.append({
                    'Year': year, 'Source': '2_proximity',
                    'Metric': f'{strategy}_MedianDistPx',
                    'Mean': float(sub['MedianDistPx'].mean()),
                    'Std': float(sub['MedianDistPx'].std(ddof=0)),
                    'N': len(sub),
                })
                summary_rows.append({
                    'Year': year, 'Source': '2_proximity',
                    'Metric': f'{strategy}_FracWithin160Px',
                    'Mean': float(sub['FracWithin160Px'].mean()),
                    'Std': float(sub['FracWithin160Px'].std(ddof=0)),
                    'N': len(sub),
                })

        r3 = df_link3[df_link3['Year'].astype(str) == y]
        if not r3.empty:
            summary_rows.append({
                'Year': year, 'Source': '3_feature_smoothness',
                'Metric': 'R_Spatial_Feature',
                'Mean': float(r3['R_Spatial_Feature'].iloc[0]),
                'Std': 0.0, 'N': 1,
            })

        if not df_link4.empty:
            l4 = df_link4[df_link4['Year'].astype(str) == y]
            for _, r in l4.iterrows():
                summary_rows.append({
                    'Year': year, 'Source': '4_testonly_gap',
                    'Metric': f'Gap_{r["Facies"]}',
                    'Mean': float(r['Gap']),
                    'Std': 0.0, 'N': int(r['NSeeds']),
                })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(LEAKAGE_ROOT, "summary.csv"), index=False)
    print(f"Saved: summary.csv")

    print("\n" + "-" * 80)
    print(f"{'Year':<6} {'Source':<22} {'Metric':<35} {'Mean':<12} {'N':<3}")
    print("-" * 80)
    for _, r in df_summary.iterrows():
        print(f"{r['Year']:<6} {r['Source']:<22} {r['Metric']:<35} "
              f"{r['Mean']:<12.4f} {int(r['N']):<3}")

    print(f"\nAll outputs in: {LEAKAGE_ROOT}")


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Structural diagnostics for separating spatial leakage and coverage problem."
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

    run_analysis(years, strategies, seeds)