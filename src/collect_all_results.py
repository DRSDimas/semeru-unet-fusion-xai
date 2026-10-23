"""
collect_all_results.py

Aggregates all evaluation metrics, structural diagnostics, and SHAP outputs into a single Markdown report.
Reference: Subsections 3.8, 3.9, 3.10, and 3.11.
"""

import os
import argparse
import pandas as pd
from datetime import datetime

from src.config import (
    EXPERIMENTS_ROOT,
    YEARS,
    STRATEGIES,
    SEEDS,
    get_model_name,
)


DEFAULT_OUTPUT = "./ALL_RESULTS.md"
SHAP_TOP_N = 5


# --- Helper Functions ---
def read_text(path):
    """
    Reads a text file with fallback encodings.
    """
    if not os.path.exists(path):
        return None
    for enc in ('utf-8', 'cp1252', 'latin-1', 'iso-8859-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def read_csv(path):
    """
    Reads a CSV file into a Pandas DataFrame.
    """
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        return f"__error__: {e}"


def emit_text_block(lines, txt, max_lines=None):
    """
    Appends a formatted text block to the output lines.
    """
    if txt is None:
        return False
    body = txt.rstrip()
    if max_lines is not None:
        parts = body.split("\n")
        if len(parts) > max_lines:
            body = "\n".join(parts[:max_lines]) + \
                   f"\n... ({len(parts) - max_lines} more lines)"
    lines.append("```text")
    lines.append(body)
    lines.append("```")
    return True


def emit_csv_block(lines, path, max_rows=None):
    """
    Appends a formatted CSV block to the output lines.
    """
    df = read_csv(path)
    if df is None:
        return False
    if isinstance(df, str) and df.startswith("__error__"):
        lines.append(f"*Failed to read: {df[10:]}*")
        return False
    total = len(df)
    if max_rows is not None and total > max_rows:
        df = df.head(max_rows)
        footer = f"\n# ... and {total - max_rows} more rows"
    else:
        footer = ""
    lines.append("```csv")
    lines.append(df.to_csv(index=False).rstrip() + footer)
    lines.append("```")
    return True


def section(lines, title, level=2):
    """
    Appends a Markdown section header.
    """
    lines.append("")
    lines.append(f"{'#' * level} {title}")
    lines.append("")


def subsection(lines, title, level=3):
    """
    Appends a Markdown subsection header.
    """
    lines.append("")
    lines.append(f"{'#' * level} {title}")
    lines.append("")


def emit_fig_ref(lines, path):
    """
    Appends a Markdown figure reference if the file exists.
    """
    if os.path.exists(path):
        lines.append(f"*Figure: `{path}`*")
        lines.append("")


def fmt_ms(mean, std, decimals=4):
    """
    Formats a mean and standard deviation string.
    """
    if pd.isna(mean):
        return "N/A"
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def emit_missing(lines, script):
    """
    Appends a warning message when a required output is missing.
    """
    lines.append(f"*File not found. Run `{script}`.*")
    lines.append("")


# --- Report Sections ---
def write_overview(lines, root):
    """
    Writes the introductory overview of the experiment.
    """
    lines.append("# All Results — Semeru Eruption Mapping")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("- **Architecture:** Gating Dual Encoder U-Net (S1 + S2 fusion)")
    lines.append("- **Events:** Semeru 2021 (3-class) and Semeru 2022 (binary)")
    lines.append("- **Strategies:** random split vs facies-based blocking split")
    lines.append("- **Seeds per (year, strategy):** 5")
    lines.append("- **Total models:** 20")
    lines.append("")
    lines.append("### Evaluation scopes")
    lines.append("")
    lines.append("- **full-AOI** — all valid pixels in the AOI")
    lines.append("- **test-only** — only pixels covered by a model's test patches")
    lines.append("")
    lines.append("### Section layout")
    lines.append("")
    lines.append("- **Summary metrics** — headline numbers per (year, strategy)")
    lines.append("- **Q1** — model performance tables")
    lines.append("- **Q2** — split-structure and per-facies diagnostics")
    lines.append("- **Q3** — SHAP feature attribution")
    lines.append("- **Appendix** — per-model training summaries")
    lines.append("")
    lines.append(f"- **Experiments root:** `{root}`")
    lines.append(f"- *Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("---")


def write_summary_metrics(lines, root):
    """
    Writes the aggregated summary metrics tables.
    """
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    section(lines, "Summary Metrics (mean ± std across 5 seeds)")

    # Full-AOI
    subsection(lines, "Full-AOI")
    summary_path = os.path.join(root, "results", "summary_metrics.csv")
    if not os.path.exists(summary_path):
        emit_missing(lines, "evaluate_all_metrics.py")
    else:
        df = pd.read_csv(summary_path)
        lines.append("| Year | Strategy | N | Native mIoU | Native Macro F1 | "
                     "Binary mIoU | Deposit IoU |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            lines.append(
                f"| {row['Year']} | {row['Strategy']} | {int(row['N'])} | "
                f"{fmt_ms(row.get('mIoU_mean'), row.get('mIoU_std'))} | "
                f"{fmt_ms(row.get('Macro_F1_mean'), row.get('Macro_F1_std'))} | "
                f"{fmt_ms(row.get('Bin_mIoU_mean'), row.get('Bin_mIoU_std'))} | "
                f"{fmt_ms(row.get('Bin_Deposit_IoU_mean'), row.get('Bin_Deposit_IoU_std'))} |"
            )
        lines.append("")

    # Test-only
    subsection(lines, "Test-only")
    summary_test_path = os.path.join(root, "results", "summary_metrics_testonly.csv")
    if not os.path.exists(summary_test_path):
        emit_missing(lines, "evaluate_all_metrics.py")
    else:
        df = pd.read_csv(summary_test_path)
        lines.append("| Year | Strategy | N | Native mIoU | Native Macro F1 | "
                     "Binary mIoU | Deposit IoU |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            lines.append(
                f"| {row['Year']} | {row['Strategy']} | {int(row['N'])} | "
                f"{fmt_ms(row.get('mIoU_mean'), row.get('mIoU_std'))} | "
                f"{fmt_ms(row.get('Macro_F1_mean'), row.get('Macro_F1_std'))} | "
                f"{fmt_ms(row.get('Bin_mIoU_mean'), row.get('Bin_mIoU_std'))} | "
                f"{fmt_ms(row.get('Bin_Deposit_IoU_mean'), row.get('Bin_Deposit_IoU_std'))} |"
            )
        lines.append("")

    # Per-seed test-only
    per_seed = os.path.join(root, "results", "per_seed_metrics_testonly.csv")
    if os.path.exists(per_seed):
        subsection(lines, "Per-seed metrics (test-only)")
        df = pd.read_csv(per_seed)
        cols = ['Year', 'Strategy', 'Seed', 'NumClasses', 'mIoU', 'Macro_F1',
                'Bin_Deposit_IoU']
        cols = [c for c in cols if c in df.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")


def write_q1(lines, root):
    """
    Writes the model performance tables (Ensemble, Facies, Training Summaries).
    """
    # See Subsection 3.9.2 (Ensemble) and 3.9.3 (Evaluation)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Q1 — Model Performance")
    lines.append("")

    # Ensemble
    subsection(lines, "Ensemble metrics (5-seed)")
    ensemble_csv = os.path.join(root, "ensemble", "ensemble_metrics.csv")
    if os.path.exists(ensemble_csv):
        df = pd.read_csv(ensemble_csv)
        lines.append("| Year | Strategy | Native mIoU | Binary mIoU | "
                     "Deposit IoU | Mean agreement | Fraction unanimous |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | "
                f"{r['mIoU']:.4f} | {r['Bin_mIoU']:.4f} | "
                f"{r['Bin_Deposit_IoU']:.4f} | "
                f"{r.get('MeanAgreementOnDeposit', float('nan')):.2f} "
                f"/ {int(r.get('NSeeds', 5))} | "
                f"{r.get('FractionDepositUnanimous', float('nan')):.3f} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "ensemble_maps.py")

    subsection(lines, "Ensemble figures")
    fig_dir = os.path.join(root, "ensemble", "figures")
    emit_fig_ref(lines, os.path.join(fig_dir, "ensemble_comparison_grid.png"))
    emit_fig_ref(lines, os.path.join(fig_dir, "agreement_comparison_grid.png"))

    # Per-facies full-AOI
    subsection(lines, "Per-facies mIoU (full-AOI)")
    facies_path = os.path.join(root, "results", "facies_metrics.csv")
    if os.path.exists(facies_path):
        df = pd.read_csv(facies_path)
        agg = (df.groupby(['Year', 'Strategy', 'Facies'])[['mIoU', 'Macro_F1']]
                 .agg(['mean', 'std']).reset_index())
        agg.columns = ['Year', 'Strategy', 'Facies',
                       'mIoU_mean', 'mIoU_std', 'F1_mean', 'F1_std']
        lines.append("| Year | Strategy | Facies | mIoU | Macro F1 |")
        lines.append("|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | {r['Facies']} | "
                f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f} | "
                f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "evaluate_all_metrics.py")

    # Per-facies test-only
    subsection(lines, "Per-facies mIoU (test-only)")
    facies_test = os.path.join(root, "results", "facies_metrics_testonly.csv")
    if os.path.exists(facies_test):
        df = pd.read_csv(facies_test)
        agg = (df.groupby(['Year', 'Strategy', 'Facies'])
                 [['mIoU', 'Macro_F1', 'NPixels']]
                 .agg(mIoU_mean=('mIoU', 'mean'),
                      mIoU_std=('mIoU', 'std'),
                      F1_mean=('Macro_F1', 'mean'),
                      F1_std=('Macro_F1', 'std'),
                      NPix_mean=('NPixels', 'mean'))
                 .reset_index())
        lines.append("| Year | Strategy | Facies | mIoU | Macro F1 | NPixels |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            npix = int(r['NPix_mean']) if pd.notna(r['NPix_mean']) else 0
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | {r['Facies']} | "
                f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f} | "
                f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f} | {npix:,} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "evaluate_all_metrics.py")

    # Training summaries
    subsection(lines, "Training summaries (per model)")
    lines.append("| Model | Best Epoch | Best Val Loss | Test mIoU | Test F1 |")
    lines.append("|---|---|---|---|---|")

    for year in YEARS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                name = get_model_name(year, strategy, seed)
                path = os.path.join(root, name, "summary.txt")
                if not os.path.exists(path):
                    continue
                txt = read_text(path)
                best_ep = best_loss = test_iou = test_f1 = "N/A"
                for line in txt.splitlines():
                    if line.startswith("Best epoch:"):
                        best_ep = line.split(":", 1)[1].strip()
                    elif line.startswith("Best val loss:"):
                        best_loss = line.split(":", 1)[1].strip()
                    elif line.startswith("Test mIoU:"):
                        test_iou = line.split(":", 1)[1].strip()
                    elif line.startswith("Test Macro F1:"):
                        test_f1 = line.split(":", 1)[1].strip()
                lines.append(
                    f"| {name} | {best_ep} | {best_loss} | "
                    f"{test_iou} | {test_f1} |"
                )
    lines.append("")


def write_q2(lines, root):
    """
    Writes the structural diagnostics tables.
    """
    # See Subsection 3.10 (Structural Diagnostics for Mechanism Separation)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Q2 — Split-Structure Diagnostics")
    lines.append("")

    leakage_dir = os.path.join(root, "leakage")

    # Link 1
    subsection(lines, "Link 1 — Label autocorrelation")
    l1 = os.path.join(leakage_dir, "link1_label_autocorrelation.csv")
    if os.path.exists(l1):
        df = pd.read_csv(l1)
        lines.append("| Year | Fraction adjacent patches sharing label | "
                     "Moran's I on label map | N valid pixels |")
        lines.append("|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Year']} | "
                f"{r['FractionAdjacentSharingLabel']:.4f} | "
                f"{r['MoransI_LabelMap']:.4f} | "
                f"{int(r['NValidPixels']):,} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    # Link 2
    subsection(lines, "Link 2 — Train-test proximity")
    l2 = os.path.join(leakage_dir, "link2_proximity_stats.csv")
    if os.path.exists(l2):
        df = pd.read_csv(l2)
        agg = (df.groupby(['Year', 'Strategy'])
                 .agg(MedianDistPx_mean=('MedianDistPx', 'mean'),
                      MedianDistPx_std=('MedianDistPx', 'std'),
                      MeanDistPx_mean=('MeanDistPx', 'mean'),
                      MeanDistPx_std=('MeanDistPx', 'std'),
                      Frac160_mean=('FracWithin160Px', 'mean'),
                      Frac160_std=('FracWithin160Px', 'std'),
                      Frac80_mean=('FracWithin80Px', 'mean'),
                      Frac80_std=('FracWithin80Px', 'std'))
                 .reset_index())
        lines.append("| Year | Strategy | Median dist (px) | Mean dist (px) | "
                     "Fraction within 160 px | Fraction within 80 px |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | "
                f"{r['MedianDistPx_mean']:.0f} ± {r['MedianDistPx_std']:.0f} | "
                f"{r['MeanDistPx_mean']:.0f} ± {r['MeanDistPx_std']:.0f} | "
                f"{r['Frac160_mean']:.3f} ± {r['Frac160_std']:.3f} | "
                f"{r['Frac80_mean']:.3f} ± {r['Frac80_std']:.3f} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    emit_fig_ref(lines, os.path.join(leakage_dir, "figures",
                                      "link2_proximity_histograms.png"))

    # Link 3
    subsection(lines, "Link 3 — Feature smoothness")
    l3 = os.path.join(leakage_dir, "link3_feature_smoothness.csv")
    if os.path.exists(l3):
        df = pd.read_csv(l3)
        lines.append("| Year | N pairs | r(spatial, feature) | "
                     "r(spatial, label disagree) | "
                     "r(feature, label disagree) |")
        lines.append("|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Year']} | {int(r['NPairs']):,} | "
                f"{r['R_Spatial_Feature']:+.4f} | "
                f"{r['R_Spatial_LabelDisagree']:+.4f} | "
                f"{r['R_Feature_LabelDisagree']:+.4f} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    # Link 4
    subsection(lines, "Link 4 — Test-only per-facies gap")
    l4 = os.path.join(leakage_dir, "link4_gap_concentration.csv")
    if os.path.exists(l4):
        df = pd.read_csv(l4)
        lines.append("| Year | Facies | Random mIoU | Random NPixels | "
                     "Blocking mIoU | Blocking NPixels | Gap (R − B) | NSeeds |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Facies']} | "
                f"{r['Random_mIoU']:.4f} ± {r['Random_Std']:.4f} | "
                f"{int(r['Random_NPixels']):,} | "
                f"{r['Blocking_mIoU']:.4f} ± {r['Blocking_Std']:.4f} | "
                f"{int(r['Blocking_NPixels']):,} | "
                f"{r['Gap']:+.4f} | {int(r['NSeeds'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    emit_fig_ref(lines, os.path.join(leakage_dir, "figures",
                                      "link4_gap_concentration.png"))

    # Full-AOI per-facies
    subsection(lines, "Full-AOI per-facies mIoU")
    df_full = os.path.join(leakage_dir, "q2_discussion_fasies_fullAOI.csv")
    if os.path.exists(df_full):
        df = pd.read_csv(df_full)
        lines.append("| Year | Strategy | Facies | mIoU | Macro F1 | NSeeds |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | {r['Facies']} | "
                f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f} | "
                f"{r['Macro_F1_mean']:.4f} | {int(r['NSeeds'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    # Overprediction
    subsection(lines, "Per-facies overprediction (full-AOI)")
    df_over = os.path.join(leakage_dir, "q2_discussion_fasies_overprediction.csv")
    if os.path.exists(df_over):
        df = pd.read_csv(df_over)
        agg = (df.groupby(['Year', 'Strategy', 'Facies'])
                 .agg(NSeeds=('Seed', 'count'),
                      FPRate_mean=('FPRate', 'mean'),
                      FPRate_std=('FPRate', 'std'),
                      Precision_mean=('DepositPrecision', 'mean'),
                      Precision_std=('DepositPrecision', 'std'))
                 .reset_index())
        lines.append("| Year | Strategy | Facies | BG→deposit FP rate | "
                     "Deposit precision | N seeds |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | {r['Facies']} | "
                f"{r['FPRate_mean']:.4f} ± {r['FPRate_std']:.4f} | "
                f"{r['Precision_mean']:.4f} ± {r['Precision_std']:.4f} | "
                f"{int(r['NSeeds'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    emit_fig_ref(lines, os.path.join(leakage_dir, "figures",
                                      "q2_fasies_overprediction.png"))

    # Gap shrinkage
    subsection(lines, "Paired-year gap (random − blocking)")
    df_gap = os.path.join(leakage_dir, "q2_discussion_gap_shrinkage.csv")
    if os.path.exists(df_gap):
        df = pd.read_csv(df_gap)
        lines.append("| Scope | Year | Random native mIoU | Blocking native mIoU | "
                     "Gap native | Gap binary | Gap deposit IoU |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(
                f"| {r['Scope']} | {r['Year']} | "
                f"{r['Random_native_mIoU']:.4f} | "
                f"{r['Blocking_native_mIoU']:.4f} | "
                f"{r['Gap_native']:+.4f} | "
                f"{r['Gap_binary']:+.4f} | "
                f"{r['Gap_deposit_IoU']:+.4f} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")

    # Summary CSV
    subsection(lines, "Summary (long-format)")
    s = os.path.join(leakage_dir, "summary.csv")
    if os.path.exists(s):
        emit_csv_block(lines, s)
        lines.append("")
    else:
        emit_missing(lines, "leakage_analysis.py")


def read_shap_top_bands(root, year, strategy, seed, top_n):
    """
    Reads the top N bands from the feature importance CSV.
    """
    # See Subsection 3.11.2 (Quantitative Aggregation Metrics)
    path = os.path.join(
        root, get_model_name(year, strategy, seed),
        "shap_gradient", "test", "feature_importance.csv"
    )
    df = read_csv(path)
    if df is None or isinstance(df, str):
        return None
    if 'Band' not in df.columns or 'Combined' not in df.columns:
        return None
    df = df.sort_values('Combined', ascending=False).head(top_n)
    return df['Band'].tolist()


def write_q3(lines, root):
    """
    Writes the SHAP feature attribution stability tables.
    """
    # See Subsection 3.11 (Feature Attribution with GradientSHAP)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Q3 — SHAP Feature Attribution")
    lines.append("")

    shap_root = os.path.join(root, "shap_analysis")

    # Band-level rankings
    subsection(lines, f"Top-{SHAP_TOP_N} bands per seed (by mean |SHAP|)")
    for year in YEARS:
        for strategy in STRATEGIES:
            lines.append(f"**{year} — {strategy}**")
            lines.append("")
            header = "| Seed | " + " | ".join(
                [f"Rank {i+1}" for i in range(SHAP_TOP_N)]) + " |"
            sep = "|" + "---|" * (SHAP_TOP_N + 1)
            lines.append(header)
            lines.append(sep)

            any_rows = False
            for seed in SEEDS:
                bands = read_shap_top_bands(root, year, strategy, seed,
                                             SHAP_TOP_N)
                if bands is None:
                    continue
                lines.append(f"| {seed} | " + " | ".join(bands) + " |")
                any_rows = True

            if not any_rows:
                lines.append(f"| _none_ | " +
                             " | ".join(["-"] * SHAP_TOP_N) + " |")
            lines.append("")

    # Cross-seed stability
    subsection(lines, "Cross-seed feature ranking stability")
    cs = os.path.join(shap_root, "cross_seed_stability.csv")
    if os.path.exists(cs):
        df = pd.read_csv(cs)
        agg = (df.groupby(['Year', 'Strategy'])['Spearman']
                 .agg(['mean', 'std', 'count']).reset_index())
        lines.append("| Year | Strategy | Mean Spearman rho | Std | N pairs |")
        lines.append("|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "shap_stability.py")

    for year in YEARS:
        emit_fig_ref(lines, os.path.join(shap_root, "figures",
                                          f"cross_seed_rank_corr_{year}.png"))
        for strategy in STRATEGIES:
            emit_fig_ref(lines, os.path.join(
                shap_root, "figures",
                f"top_bands_per_seed_{year}_{strategy}.png"))

    # Facies invariance
    subsection(lines, "Fasies-invariance of feature rankings")
    fi = os.path.join(shap_root, "fasies_invariance.csv")
    if os.path.exists(fi):
        df = pd.read_csv(fi)
        agg = (df.groupby(['Year', 'Strategy'])['rank_std']
                 .agg(['mean', 'std', 'count']).reset_index())
        lines.append("| Year | Strategy | Mean rank std | Std | N obs |")
        lines.append("|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "shap_stability.py")

    for year in YEARS:
        for strategy in STRATEGIES:
            emit_fig_ref(lines, os.path.join(
                shap_root, "figures",
                f"fasies_invariance_{year}_{strategy}.png"))

    # Modality
    subsection(lines, "Modality reliance (S1_frac) per seed")
    mod = os.path.join(shap_root, "modality_by_seed.csv")
    if os.path.exists(mod):
        df = pd.read_csv(mod)
        agg = (df.groupby(['Year', 'Strategy', 'Class'])['S1_frac']
                 .agg(['mean', 'std', 'count']).reset_index())
        lines.append("| Year | Strategy | Class | Mean S1_frac | Std | N |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['Year']} | {r['Strategy']} | {r['Class']} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |"
            )
        lines.append("")
    else:
        emit_missing(lines, "shap_stability.py")

    for year in YEARS:
        emit_fig_ref(lines, os.path.join(shap_root, "figures",
                                          f"modality_by_seed_{year}.png"))

    # Mechanism link
    subsection(lines, "Per-seed stability vs performance")
    mech = os.path.join(shap_root, "mechanism_vs_iou.csv")
    if os.path.exists(mech):
        emit_csv_block(lines, mech)
        lines.append("")
    else:
        emit_missing(lines, "shap_stability.py")

    for year in YEARS:
        emit_fig_ref(lines, os.path.join(shap_root, "figures",
                                          f"stability_vs_iou_{year}.png"))

    # Summary
    subsection(lines, "SHAP summary (long-format)")
    s = os.path.join(shap_root, "summary.csv")
    if os.path.exists(s):
        emit_csv_block(lines, s)
        lines.append("")
    else:
        emit_missing(lines, "shap_stability.py")


def write_appendix(lines, root):
    """
    Writes the per-model training summary outputs.
    """
    # See Subsection 3.8 (Model Training Configuration)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("# Appendix — Per-Model Training Summaries")
    lines.append("")

    for year in YEARS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                name = get_model_name(year, strategy, seed)
                path = os.path.join(root, name, "summary.txt")
                subsection(lines, name)
                if not emit_text_block(lines, read_text(path)):
                    lines.append(f"*Missing: `{path}`*")
                    lines.append("")


# --- Main Execution ---
def main():
    """
    Aggregates outputs into a single Markdown report.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate experiment outputs into a Markdown data dump."
    )
    parser.add_argument("--root", type=str, default=EXPERIMENTS_ROOT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root = args.root
    output = args.output

    print(f"Collecting from: {root}")
    print(f"Writing to:      {output}")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Experiments root not found: {root}")

    lines = []
    write_overview(lines, root)
    write_summary_metrics(lines, root)
    write_q1(lines, root)
    write_q2(lines, root)
    write_q3(lines, root)
    write_appendix(lines, root)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"Report saved: {output}")
    print(f"Total lines:  {len(lines)}")


if __name__ == "__main__":
    main()