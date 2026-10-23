"""
shap_stability.py

Computes feature-attribution stability diagnostics (cross-seed, facies, and modality).
Reference: Subsection 3.11.3 (Attribution Stability Testing).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr

from .config import (
    get_all_configs,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    get_shap_dir,
    get_class_names,
    EXPERIMENTS_ROOT,
    FACIES_NAMES,
)


# --- Constants ---
SHAP_ANALYSIS_ROOT = os.path.join(EXPERIMENTS_ROOT, "shap_analysis")
FIGURES_DIR = os.path.join(SHAP_ANALYSIS_ROOT, "figures")
PER_SEED_METRICS = os.path.join(EXPERIMENTS_ROOT, "results", "per_seed_metrics.csv")


# --- Loaders ---
def load_feature_importance(year, strategy, seed):
    """
    Loads the global feature importance CSV for a given model.
    """
    # See Subsection 3.11.3
    path = os.path.join(get_shap_dir(year, strategy, seed, "test"),
                        "feature_importance.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Combined' not in df.columns:
        return None
    return df[['Band', 'Combined']].copy()


def load_modality_fraction(year, strategy, seed):
    """
    Parses the modality fraction text file into a dictionary.
    """
    # See Subsection 3.11.3
    path = os.path.join(get_shap_dir(year, strategy, seed, "test"),
                        "modality_fraction.txt")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ": S1_frac = " in line:
                cname = line.split(":")[0].strip()
                val = float(line.split("= ")[1])
                out[cname] = val
    return out if out else None


def load_per_facies(year, strategy, seed):
    """
    Loads feature importance CSVs for each facies.
    """
    # See Subsection 3.11.3
    base = os.path.join(get_shap_dir(year, strategy, seed, "test"),
                        "per_facies")
    if not os.path.isdir(base):
        return None
    out = {}
    for fname in FACIES_NAMES:
        path = os.path.join(base, f"{fname}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'Combined' not in df.columns:
            continue
        out[fname] = df[['Band', 'Combined']].copy()
    return out if out else None


def load_per_seed_metrics():
    """
    Loads the evaluation metrics for all seeds.
    """
    if not os.path.exists(PER_SEED_METRICS):
        return None
    return pd.read_csv(PER_SEED_METRICS)


# --- Cross-Seed Stability ---
def compute_cross_seed_stability(year, strategy, seeds):
    """
    Computes pairwise Spearman correlations between feature rankings across seeds.
    """
    # See Subsection 3.11.3 (Equation 3.28)
    seed_imps = {}
    for seed in seeds:
        df = load_feature_importance(year, strategy, seed)
        if df is not None:
            seed_imps[seed] = df

    if len(seed_imps) < 2:
        return pd.DataFrame(), float('nan'), float('nan'), seed_imps

    # Align features across seeds
    bands = sorted(set.intersection(
        *[set(df['Band']) for df in seed_imps.values()]
    ))
    matrix = pd.DataFrame(index=bands)
    for seed, df in seed_imps.items():
        matrix[seed] = df.set_index('Band')['Combined'].reindex(bands)

    # Compute pairwise Spearman rank correlation
    rows = []
    for a, b in combinations(seed_imps.keys(), 2):
        rho, _ = spearmanr(matrix[a].values, matrix[b].values)
        rows.append({
            'Year': year,
            'Strategy': strategy,
            'SeedA': a,
            'SeedB': b,
            'Spearman': float(rho),
        })
    pairwise = pd.DataFrame(rows)

    return (pairwise,
            float(pairwise['Spearman'].mean()),
            float(pairwise['Spearman'].std(ddof=0)),
            seed_imps)


# --- Facies Invariance ---
def compute_fasies_invariance(year, strategy, seed):
    """
    Computes the variance of band rankings across the five facies.
    """
    # See Subsection 3.11.3 (Equation 3.29)
    per_facies = load_per_facies(year, strategy, seed)
    if per_facies is None or len(per_facies) < 2:
        return None

    bands = sorted(set.intersection(
        *[set(df['Band']) for df in per_facies.values()]
    ))
    if not bands:
        return None

    rank_matrix = pd.DataFrame(index=bands)
    for fname, df in per_facies.items():
        s = df.set_index('Band')['Combined'].reindex(bands)
        rank_matrix[fname] = s.rank(ascending=False)

    return pd.DataFrame({
        'Band': bands,
        'rank_mean': rank_matrix.mean(axis=1).values,
        'rank_std':  rank_matrix.std(axis=1, ddof=0).values,
    }).sort_values('rank_mean')


# --- Modality Stability ---
def compute_modality_stability(year, strategy, seeds):
    """
    Collects S1 fraction per seed per class, then computes mean and std.
    """
    # See Subsection 3.11.3
    rows = []
    for seed in seeds:
        mf = load_modality_fraction(year, strategy, seed)
        if mf is None:
            continue
        for cname, val in mf.items():
            rows.append({
                'Year': year,
                'Strategy': strategy,
                'Seed': seed,
                'Class': cname,
                'S1_frac': val,
            })
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = (df.groupby(['Year', 'Strategy', 'Class'])['S1_frac']
                 .agg(['mean', 'std', 'count'])
                 .reset_index())
    summary.columns = ['Year', 'Strategy', 'Class', 'mean', 'std', 'n']
    return df, summary


# --- Mechanism Link ---
def build_mechanism_table(year, strategy, seeds, cross_seed_df, metrics_df):
    """
    Correlates per-seed stability metrics with per-seed IoU performance.
    """
    # See Subsection 3.11.3 (Equation 3.30)
    if metrics_df is None:
        return pd.DataFrame()

    sub = metrics_df[(metrics_df['Strategy'] == strategy) &
                     (metrics_df['Year'].astype(str) == str(year))]
    if sub.empty:
        return pd.DataFrame()

    iou_col = 'Bin_Deposit_IoU'
    if iou_col not in sub.columns:
        for c in ['mIoU', 'Bin_Deposit_IoU', 'Deposit_IoU']:
            if c in sub.columns:
                iou_col = c
                break

    rows = []
    for seed in seeds:
        m = sub[sub['Seed'] == seed]
        if len(m) == 0:
            continue
        perf = float(m[iou_col].iloc[0])

        per_band = compute_fasies_invariance(year, strategy, seed)
        fasies_rank_std = (float(per_band['rank_std'].mean())
                           if per_band is not None else float('nan'))

        mf = load_modality_fraction(year, strategy, seed)
        s1_mean = float(np.mean(list(mf.values()))) if mf else float('nan')

        seed_rho = float('nan')
        if not cross_seed_df.empty:
            mask = ((cross_seed_df['SeedA'] == seed) |
                    (cross_seed_df['SeedB'] == seed))
            if mask.any():
                seed_rho = float(cross_seed_df[mask]['Spearman'].mean())

        rows.append({
            'Year': year,
            'Strategy': strategy,
            'Seed': seed,
            'CrossSeedRho': seed_rho,
            'FasiesRankStd': fasies_rank_std,
            'S1_frac_mean': s1_mean,
            'Performance': perf,
            'PerformanceMetric': iou_col,
        })

    return pd.DataFrame(rows)


# --- Visualization Functions ---
def plot_cross_seed_rank_corr(pairwise_df, year, output_path):
    if pairwise_df.empty:
        return
    strategies = sorted(pairwise_df['Strategy'].unique())
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(6 * len(strategies), 5),
                             sharey=True, squeeze=False)
    axes = axes.flatten()

    for ax, strategy in zip(axes, strategies):
        sub = pairwise_df[pairwise_df['Strategy'] == strategy]
        labels = [f"{int(a)}-{int(b)}" for a, b in
                  zip(sub['SeedA'], sub['SeedB'])]
        color = '#1f77b4' if strategy == 'random' else '#ff7f0e'
        ax.bar(range(len(sub)), sub['Spearman'], color=color, alpha=0.7)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(sub['Spearman'].mean(), color='red', linestyle='--',
                   linewidth=1, label=f"mean = {sub['Spearman'].mean():.3f}")
        ax.set_title(f"{strategy} split", fontsize=11)
        ax.set_ylabel("Spearman rho between seeds")
        ax.set_ylim(-0.2, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Cross-seed feature-ranking stability — {year}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_bands_per_seed(seed_imps, year, strategy, output_path, top_n=8):
    if not seed_imps:
        return
    seeds = sorted(seed_imps.keys())
    n = len(seeds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6), squeeze=False)
    axes = axes.flatten()

    color = '#1f77b4' if strategy == 'random' else '#ff7f0e'
    for ax, seed in zip(axes, seeds):
        df = seed_imps[seed].sort_values('Combined', ascending=False).head(top_n)
        ax.barh(range(len(df)), df['Combined'].values, color=color, alpha=0.75)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Band'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"seed {seed}", fontsize=10)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f"Top-{top_n} bands per seed — {year} {strategy}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fasies_invariance(per_band_dict, year, strategy, output_path):
    if not per_band_dict:
        return
    seeds = sorted(per_band_dict.keys())
    n = len(seeds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 7),
                             sharex=False, sharey=True, squeeze=False)
    axes = axes.flatten()

    color = '#1f77b4' if strategy == 'random' else '#ff7f0e'
    for ax, seed in zip(axes, seeds):
        df = per_band_dict[seed].sort_values('rank_mean').head(12)
        ax.barh(range(len(df)), df['rank_std'].values,
                color=color, alpha=0.75)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Band'].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"seed {seed}", fontsize=10)
        ax.set_xlabel("Rank std across facies", fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(
        f"Per-band ranking variance across facies — {year} {strategy}\n"
        f"(higher = more location-specific; lower = more invariant)",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_modality_by_seed(modality_df, year, output_path):
    if modality_df.empty:
        return
    strategies = sorted(modality_df['Strategy'].unique())
    classes = sorted(modality_df['Class'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    positions, labels, colors, data = [], [], [], []
    pos = 1
    for c in classes:
        for s in strategies:
            vals = modality_df[(modality_df['Class'] == c) &
                               (modality_df['Strategy'] == s)]['S1_frac'].values
            if len(vals) == 0:
                continue
            positions.append(pos)
            labels.append(f"{c}\n{s}")
            colors.append('#1f77b4' if s == 'random' else '#ff7f0e')
            data.append(vals)
            pos += 1
        pos += 0.5

    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for p, vals in zip(positions, data):
        ax.scatter([p] * len(vals), vals, color='black', s=20, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("S1 fraction of |SHAP|")
    ax.set_title(f"Modality reliance by seed — {year}")
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_stability_vs_iou(mech_df, year, output_path):
    if mech_df.empty:
        return
    strategies = sorted(mech_df['Strategy'].unique())
    colors_map = {'random': '#1f77b4', 'blocking': '#ff7f0e'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Cross-seed stability vs Performance
    ax = axes[0]
    for s in strategies:
        sub = mech_df[mech_df['Strategy'] == s]
        ax.scatter(sub['CrossSeedRho'], sub['Performance'],
                   color=colors_map.get(s, 'gray'), label=s, s=80, alpha=0.8)
        for _, row in sub.iterrows():
            ax.annotate(str(int(row['Seed'])),
                        (row['CrossSeedRho'], row['Performance']),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel("Cross-seed stability (mean Spearman rho)")
    ax.set_ylabel(f"Per-seed {mech_df['PerformanceMetric'].iloc[0]}")
    ax.set_title("Cross-seed stability vs performance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel B: Facies invariance vs Performance
    ax = axes[1]
    for s in strategies:
        sub = mech_df[mech_df['Strategy'] == s]
        ax.scatter(sub['FasiesRankStd'], sub['Performance'],
                   color=colors_map.get(s, 'gray'), label=s, s=80, alpha=0.8)
        for _, row in sub.iterrows():
            ax.annotate(str(int(row['Seed'])),
                        (row['FasiesRankStd'], row['Performance']),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel("Fasies rank std (lower = more invariant)")
    ax.set_ylabel(f"Per-seed {mech_df['PerformanceMetric'].iloc[0]}")
    ax.set_title("Fasies-invariance vs performance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Feature stability vs performance — {year}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Main Execution ---
def run_analysis(years, strategies, seeds):
    """
    Orchestrates the stability analysis pipeline.
    """
    # See Subsection 3.11.3
    os.makedirs(SHAP_ANALYSIS_ROOT, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics_df = load_per_seed_metrics()
    if metrics_df is None:
        print(f"WARNING: {PER_SEED_METRICS} not found. "
              f"Mechanism link will be skipped.")

    all_cross_seed = []
    all_fasies = []
    all_modality = []
    all_mechanism = []
    summary_rows = []

    for year in years:
        for strategy in strategies:
            print(f"\n{'='*70}")
            print(f"  SHAP STABILITY: {year} {strategy}")
            print(f"{'='*70}")

            # 1. Cross-seed stability
            pairwise, mean_rho, std_rho, seed_imps = \
                compute_cross_seed_stability(year, strategy, seeds)

            if not pairwise.empty:
                all_cross_seed.append(pairwise)
                print(f"  Cross-seed stability: mean rho = "
                      f"{mean_rho:.3f} +/- {std_rho:.3f} "
                      f"({len(pairwise)} pairs)")
                summary_rows.append({
                    'Year': year,
                    'Strategy': strategy,
                    'Metric': 'cross_seed_spearman',
                    'Mean': mean_rho,
                    'Std': std_rho,
                    'N': len(pairwise),
                })
                plot_top_bands_per_seed(
                    seed_imps, year, strategy,
                    os.path.join(FIGURES_DIR,
                                 f"top_bands_per_seed_{year}_{strategy}.png")
                )

            # 2. Facies invariance
            per_band_dict = {}
            for seed in seeds:
                per_band = compute_fasies_invariance(year, strategy, seed)
                if per_band is not None:
                    per_band_dict[seed] = per_band
                    for _, r in per_band.iterrows():
                        all_fasies.append({
                            'Year': year,
                            'Strategy': strategy,
                            'Seed': seed,
                            'Band': r['Band'],
                            'rank_mean': r['rank_mean'],
                            'rank_std': r['rank_std'],
                        })

            if per_band_dict:
                mean_std = float(np.mean(
                    [df['rank_std'].mean() for df in per_band_dict.values()]
                ))
                std_std = float(np.std(
                    [df['rank_std'].mean() for df in per_band_dict.values()]
                ))
                print(f"  Facies rank std: mean = {mean_std:.3f} "
                      f"+/- {std_std:.3f}")
                summary_rows.append({
                    'Year': year,
                    'Strategy': strategy,
                    'Metric': 'fasies_rank_std',
                    'Mean': mean_std,
                    'Std': std_std,
                    'N': len(per_band_dict),
                })
                plot_fasies_invariance(
                    per_band_dict, year, strategy,
                    os.path.join(FIGURES_DIR,
                                 f"fasies_invariance_{year}_{strategy}.png")
                )

            # 3. Modality reliance
            mod_df, mod_summary = compute_modality_stability(year, strategy, seeds)
            if not mod_df.empty:
                all_modality.append(mod_df)
                for _, r in mod_summary.iterrows():
                    summary_rows.append({
                        'Year': year,
                        'Strategy': strategy,
                        'Metric': f"s1_frac_{r['Class']}",
                        'Mean': float(r['mean']),
                        'Std': float(r['std']),
                        'N': int(r['n']),
                    })
                    print(f"  S1_frac [{r['Class']}]: "
                          f"{r['mean']:.3f} +/- {r['std']:.3f}")

            # 4. Mechanism link
            mech = build_mechanism_table(year, strategy, seeds,
                                         pairwise, metrics_df)
            if not mech.empty:
                all_mechanism.append(mech)

    # Save CSVs
    if all_cross_seed:
        pd.concat(all_cross_seed, ignore_index=True).to_csv(
            os.path.join(SHAP_ANALYSIS_ROOT, "cross_seed_stability.csv"),
            index=False)
        print(f"\nSaved: cross_seed_stability.csv")

    if all_fasies:
        pd.DataFrame(all_fasies).to_csv(
            os.path.join(SHAP_ANALYSIS_ROOT, "fasies_invariance.csv"),
            index=False)
        print(f"Saved: fasies_invariance.csv")

    if all_modality:
        pd.concat(all_modality, ignore_index=True).to_csv(
            os.path.join(SHAP_ANALYSIS_ROOT, "modality_by_seed.csv"),
            index=False)
        print(f"Saved: modality_by_seed.csv")

    if all_mechanism:
        pd.concat(all_mechanism, ignore_index=True).to_csv(
            os.path.join(SHAP_ANALYSIS_ROOT, "mechanism_vs_iou.csv"),
            index=False)
        print(f"Saved: mechanism_vs_iou.csv")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(SHAP_ANALYSIS_ROOT, "summary.csv"), index=False)
        print(f"Saved: summary.csv")

    # Generate Figures
    if all_cross_seed:
        cs_df = pd.concat(all_cross_seed, ignore_index=True)
        for year in years:
            sub = cs_df[cs_df['Year'].astype(str) == str(year)]
            if not sub.empty:
                plot_cross_seed_rank_corr(
                    sub, year,
                    os.path.join(FIGURES_DIR,
                                 f"cross_seed_rank_corr_{year}.png"))

    if all_modality:
        mod_df = pd.concat(all_modality, ignore_index=True)
        for year in years:
            sub = mod_df[mod_df['Year'].astype(str) == str(year)]
            if not sub.empty:
                plot_modality_by_seed(
                    sub, year,
                    os.path.join(FIGURES_DIR, f"modality_by_seed_{year}.png"))

    if all_mechanism:
        mech_df = pd.concat(all_mechanism, ignore_index=True)
        for year in years:
            sub = mech_df[mech_df['Year'].astype(str) == str(year)]
            if not sub.empty:
                plot_stability_vs_iou(
                    sub, year,
                    os.path.join(FIGURES_DIR,
                                 f"stability_vs_iou_{year}.png"))

    # Console Summary
    if summary_rows:
        df_s = pd.DataFrame(summary_rows)
        print("\n" + "=" * 80)
        print("SHAP STABILITY SUMMARY")
        print("=" * 80)
        print(f"{'Year':<6} {'Strategy':<10} {'Metric':<30} "
              f"{'Mean':<12} {'Std':<10} {'N':<4}")
        print("-" * 80)
        for _, r in df_s.iterrows():
            print(f"{r['Year']:<6} {r['Strategy']:<10} {r['Metric']:<30} "
                  f"{r['Mean']:<12.4f} {r['Std']:<10.4f} {int(r['N']):<4}")

    print(f"\nAll outputs in: {SHAP_ANALYSIS_ROOT}")


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP stability analysis (cross-seed, facies, and modality)."
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