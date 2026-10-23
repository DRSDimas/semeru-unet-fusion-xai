"""
gradient_shap.py

Computes GradientSHAP feature attribution and spatial maps for trained models.
Reference: Subsections 3.11.1, 3.11.2, 3.11.3, and 3.11.4.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt

from .config import (
    get_config,
    get_all_configs,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    get_checkpoint_path,
    get_shap_dir,
    get_data_root,
    get_class_names,
    get_class_slugs,
    FACIES_NAMES,
)
from .models import get_model
from .data_loader import load_geotiffs, SemeruPatchDataset
from .xai_utils import get_band_names, create_gaussian_weight, save_geotiff


# --- Constants ---
S1_INDICES = [0, 1, 2, 3]
S2_INDICES = list(range(4, 14))
SHAP_SEED = 42


# --- Seed Initialization ---
def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Model Wrapper ---
class SoftmaxWrapper(torch.nn.Module):
    """
    Wraps the model to output per-class patch-mean probabilities.
    """
    # See Subsection 3.11.1 (Method and Configuration GradientSHAP)
    # See Equation 3.24 (Patch-level Score)
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        probs = F.softmax(self.model(x), dim=1)
        return probs.mean(dim=[2, 3])


# --- SHAP Execution Handlers ---
def _call_shap_values(explainer, X, nsamples, seed):
    """
    Executes explainer.shap_values with fallback signatures for compatibility.
    """
    # See Subsection 3.11.1
    attempts = [
        ("nsamples+rseed", lambda: explainer.shap_values(X, nsamples=nsamples, rseed=seed)),
        ("nsamples",       lambda: explainer.shap_values(X, nsamples=nsamples)),
        ("positional",     lambda: explainer.shap_values(X, nsamples)),
        ("default",        lambda: explainer.shap_values(X)),
    ]
    last_err = None
    for name, fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"All GradientExplainer signatures failed. Last: {last_err}")


def _normalize_shap_shape(shap_values, n_classes):
    """
    Normalizes SHAP output array to (n_classes, N, 14, H, W).
    """
    # See Subsection 3.11.1
    if isinstance(shap_values, list):
        arr = np.stack(shap_values, axis=0)
        if arr.shape[0] == n_classes:
            return arr
        raise ValueError(f"List of {arr.shape[0]} arrays, expected {n_classes}")

    arr = np.asarray(shap_values)
    if arr.ndim != 5:
        raise ValueError(f"Unexpected SHAP ndim: {arr.ndim}, shape {arr.shape}")

    # (N, 14, H, W, C)
    if arr.shape[-1] == n_classes and arr.shape[1] == 14:
        return np.moveaxis(arr, -1, 0)
    # (C, N, 14, H, W)
    if arr.shape[0] == n_classes and arr.shape[2] == 14:
        return arr
    # (N, C, 14, H, W)
    if arr.shape[1] == n_classes and arr.shape[2] == 14:
        return np.moveaxis(arr, 1, 0)

    raise ValueError(f"Unexpected SHAP shape: {arr.shape}, expected C={n_classes}")


# --- Data and Model Loading ---
def load_model_and_data(year, strategy, seed, scope='test',
                        patch_size=160, stride=80):
    """
    Loads the trained model and extracts the specified data split.
    """
    # See Subsection 3.11.1
    config = get_config(year, strategy, seed)
    device = config['device']

    model = get_model(config)
    ckpt_path = get_checkpoint_path(year, strategy, seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    data_dict = load_geotiffs(get_data_root(year))

    split_type_arg = 'test' if scope == 'test' else 'all'
    dataset = SemeruPatchDataset(
        data_dict=data_dict,
        split_type=split_type_arg,
        patch_size=patch_size,
        stride=stride,
        augment=False,
        split_strategy=strategy,
    )

    return model, dataset, config, device


# --- Patch Sampling ---
def sample_patches(dataset, n_background=30, n_test=None, seed=SHAP_SEED):
    """
    Selects background and test patches for SHAP computation.
    """
    # See Subsection 3.11.1
    rng = np.random.RandomState(seed)
    n_total = len(dataset)

    n_bg = min(n_background, max(1, n_total // 3))
    bg_indices = rng.choice(n_total, size=n_bg, replace=False).tolist()

    if n_test is None or n_test <= 0 or n_test >= n_total:
        test_indices = list(range(n_total))
        is_full = True
    else:
        remaining = [i for i in range(n_total) if i not in bg_indices]
        n_sample = min(n_test, len(remaining))
        test_indices = rng.choice(remaining, size=n_sample, replace=False).tolist()
        is_full = False

    print(f"   Background: {n_bg}  |  Test: {len(test_indices)}"
          f"{' (FULL)' if is_full else ''}")
    return bg_indices, test_indices, is_full


# --- GradientSHAP Computation ---
def compute_gradient_shap(model, dataset, bg_indices, test_indices,
                          device, num_classes, nsamples=50,
                          chunk_size=10, seed=SHAP_SEED):
    """
    Executes GradientSHAP on the selected patches.
    """
    # See Subsection 3.11.1
    wrapped = SoftmaxWrapper(model).to(device).eval()

    bg_patches = np.stack([dataset[i][0].numpy() for i in bg_indices])
    bg_tensor = torch.from_numpy(bg_patches).float().to(device)
    print(f"   Background tensor: {tuple(bg_tensor.shape)}")

    explainer = shap.GradientExplainer(wrapped, bg_tensor)

    n_total = len(test_indices)
    print(f"   SHAP on {n_total} patches (nsamples={nsamples}, "
          f"chunk={chunk_size}, classes={num_classes})...")

    all_shap = []
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_idx = test_indices[start:end]

        chunk_patches = np.stack([dataset[i][0].numpy() for i in chunk_idx])
        chunk_tensor = torch.from_numpy(chunk_patches).float().to(device)

        chunk_shap = _call_shap_values(explainer, chunk_tensor, nsamples, seed)
        chunk_shap = _normalize_shap_shape(chunk_shap, n_classes=num_classes)
        all_shap.append(chunk_shap)

        print(f"      Progress: {end}/{n_total}")

        del chunk_tensor, chunk_shap
        if device == 'cuda':
            torch.cuda.empty_cache()

    shap_array = np.concatenate(all_shap, axis=1)
    print(f"   Final SHAP array: {shap_array.shape}")

    # Build per-patch metadata for spatial maps
    test_meta = []
    for idx in test_indices:
        y, x = dataset.patch_coords[idx]
        _, _, cloud, facies = dataset[idx]
        facies_np = facies.numpy()
        u, c = np.unique(facies_np, return_counts=True)
        valid = u > 0
        majority_facies = int(u[valid][np.argmax(c[valid])]) if valid.sum() > 0 else 0
        test_meta.append({
            'idx': idx, 'y': y, 'x': x,
            'cloud_frac': float(1.0 - cloud.numpy().mean()),
            'facies_majority': majority_facies,
        })

    del bg_tensor, explainer, wrapped
    if device == 'cuda':
        torch.cuda.empty_cache()

    return shap_array, test_meta


# --- SHAP Aggregations ---
def compute_global_importance(shap_array):
    """
    Computes the mean absolute SHAP value per band per class.
    """
    # See Subsection 3.11.2 (Quantitative Aggregation Metrics)
    # See Equation 3.25
    return np.abs(shap_array).mean(axis=(1, 3, 4))


def compute_modality_fraction(shap_array):
    """
    Calculates the fraction of total absolute SHAP attributed to Sentinel-1.
    """
    # See Subsection 3.11.2
    # See Equation 3.26
    abs_shap = np.abs(shap_array)
    s1_total = abs_shap[:, :, S1_INDICES].sum(axis=(1, 2, 3, 4))
    s2_total = abs_shap[:, :, S2_INDICES].sum(axis=(1, 2, 3, 4))
    all_total = s1_total + s2_total
    return {
        's1_frac': (s1_total / (all_total + 1e-12)).tolist(),
        's1_total': s1_total.tolist(),
        's2_total': s2_total.tolist(),
    }


def compute_per_facies_importance(shap_array, test_meta):
    """
    Computes mean absolute SHAP values aggregated by facies.
    """
    # See Subsection 3.11.2
    result = {}
    for f_code, f_name in enumerate(FACIES_NAMES, start=1):
        mask = np.array([m['facies_majority'] == f_code for m in test_meta])
        if mask.sum() == 0:
            continue
        result[f_name] = np.abs(shap_array[:, mask]).mean(axis=(1, 3, 4))
    return result


def compute_cloud_conditional(shap_array, test_meta,
                              low_thresh=0.05, high_thresh=0.30):
    """
    Computes SHAP differences between low-cloud and high-cloud patches.
    """
    # See Subsection 3.11.2
    # See Equation 3.27
    cloud_fracs = np.array([m['cloud_frac'] for m in test_meta])
    low = cloud_fracs < low_thresh
    high = cloud_fracs > high_thresh
    result = {}
    if low.sum() > 0:
        result['low_cloud'] = np.abs(shap_array[:, low]).mean(axis=(1, 3, 4))
    if high.sum() > 0:
        result['high_cloud'] = np.abs(shap_array[:, high]).mean(axis=(1, 3, 4))
    return result, int(low.sum()), int(high.sum())


# --- Visualization Functions ---
def plot_feature_importance(importance, band_names, class_names,
                            output_path, title):
    combined = importance.mean(axis=0)
    order = np.argsort(combined)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.arange(len(band_names)), combined[order])
    ax.set_yticks(np.arange(len(band_names)))
    ax.set_yticklabels([band_names[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_importance(importance, band_names, class_names,
                          output_path, title):
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for i, (ax, cname) in enumerate(zip(axes, class_names)):
        vals = importance[i]
        order = np.argsort(vals)[::-1][:12]
        ax.barh(np.arange(len(order)), vals[order])
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([band_names[j] for j in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(cname)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_modality_comparison(modality_dict, class_names, output_path, title):
    s1_frac = np.array(modality_dict['s1_frac'])
    s2_frac = 1 - s1_frac
    fig, ax = plt.subplots(figsize=(2 + len(class_names), 5))
    x = np.arange(len(class_names))
    ax.bar(x, s1_frac, label='S1 (SAR)', color='steelblue')
    ax.bar(x, s2_frac, bottom=s1_frac, label='S2 (Optical)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Fraction of total |SHAP|')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    for i, v in enumerate(s1_frac):
        ax.text(i, v / 2, f'{v:.2f}', ha='center', va='center',
                color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cloud_conditional(cloud_dict, band_names, output_path, title):
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(band_names))
    width = 0.4
    if 'low_cloud' in cloud_dict:
        ax.bar(x - width/2, cloud_dict['low_cloud'].mean(axis=0), width,
               label='Low cloud', color='skyblue')
    if 'high_cloud' in cloud_dict:
        ax.bar(x + width/2, cloud_dict['high_cloud'].mean(axis=0), width,
               label='High cloud', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean |SHAP|')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Spatial Attribution Maps ---
def mosaic_shap_maps(shap_array, test_meta, output_dir, metadata,
                     num_classes, class_slugs, patch_size=160, sigma=64):
    """
    Mosaics SHAP maps into a full-AOI spatial grid using Gaussian weighting.
    """
    # See Subsection 3.11.4 (Spatial Attribution Map)
    H = metadata['height']
    W = metadata['width']

    weight_map = create_gaussian_weight(patch_size, sigma)
    s1_acc = np.zeros((num_classes, H, W), dtype=np.float64)
    s2_acc = np.zeros((num_classes, H, W), dtype=np.float64)
    all_acc = np.zeros((num_classes, H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for i, meta in enumerate(test_meta):
        y, x = meta['y'], meta['x']
        abs_shap = np.abs(shap_array[:, i])
        s1_patch = abs_shap[:, S1_INDICES].sum(axis=1)
        s2_patch = abs_shap[:, S2_INDICES].sum(axis=1)
        all_patch = abs_shap.mean(axis=1)
        for c in range(num_classes):
            s1_acc[c, y:y+patch_size, x:x+patch_size] += s1_patch[c] * weight_map
            s2_acc[c, y:y+patch_size, x:x+patch_size] += s2_patch[c] * weight_map
            all_acc[c, y:y+patch_size, x:x+patch_size] += all_patch[c] * weight_map
        weight_sum[y:y+patch_size, x:x+patch_size] += weight_map

    weight_sum = np.maximum(weight_sum, 1e-8)
    s1_acc /= weight_sum
    s2_acc /= weight_sum
    all_acc /= weight_sum

    for c, slug in enumerate(class_slugs):
        save_geotiff(s1_acc[c].astype(np.float32),
                     os.path.join(output_dir, f's1_total_shap_{slug}.tif'), metadata)
        save_geotiff(s2_acc[c].astype(np.float32),
                     os.path.join(output_dir, f's2_total_shap_{slug}.tif'), metadata)
        s1_frac = s1_acc[c] / (s1_acc[c] + s2_acc[c] + 1e-12)
        save_geotiff(s1_frac.astype(np.float32),
                     os.path.join(output_dir, f's1_fraction_{slug}.tif'), metadata)
        save_geotiff(all_acc[c].astype(np.float32),
                     os.path.join(output_dir, f'combined_shap_{slug}.tif'), metadata)

    mask = (weight_sum < 1e-6).astype(np.uint8)
    save_geotiff(mask, os.path.join(output_dir, 'test_mask.tif'),
                 metadata, dtype='uint8')


# --- Main Execution ---
def run_gradient_shap(year, strategy, seed, scope='test',
                      n_background=30, n_test=-1, nsamples=50):
    """
    Runs the full GradientSHAP pipeline for a single model.
    """
    tag = f"{year}_{strategy}_seed{seed} [{scope}]"
    print("\n" + "=" * 80)
    print(f"GRADIENTSHAP: {tag}")
    print("=" * 80)

    set_torch_seed(seed)

    output_dir = get_shap_dir(year, strategy, seed, scope)
    os.makedirs(output_dir, exist_ok=True)

    band_names = get_band_names()
    class_names = get_class_names(year)
    class_slugs = get_class_slugs(year)

    model, dataset, config, device = load_model_and_data(
        year, strategy, seed, scope=scope,
    )
    num_classes = config['num_classes']
    print(f"   Classes:     {num_classes}  ({class_names})")
    print(f"   Total patches in scope: {len(dataset)}")
    print(f"   Device:      {device}")

    bg_idx, test_idx, is_full = sample_patches(
        dataset, n_background, n_test, seed=SHAP_SEED
    )

    shap_array, test_meta = compute_gradient_shap(
        model, dataset, bg_idx, test_idx, device, num_classes,
        nsamples=nsamples, seed=SHAP_SEED,
    )

    # 1. Feature importance
    # See Subsection 3.11.2
    importance = compute_global_importance(shap_array)
    rows = {'Band': band_names}
    for c, cname in enumerate(class_names):
        rows[cname] = importance[c]
    rows['Combined'] = importance.mean(axis=0)
    pd.DataFrame(rows).sort_values('Combined', ascending=False).to_csv(
        os.path.join(output_dir, 'feature_importance.csv'), index=False
    )
    plot_feature_importance(importance, band_names, class_names,
                            os.path.join(output_dir, 'feature_importance.png'),
                            f'Feature Importance — {tag}')
    plot_class_importance(importance, band_names, class_names,
                          os.path.join(output_dir, 'class_importance.png'),
                          f'Class-wise Importance — {tag}')

    # 2. Modality fraction
    # See Subsection 3.11.2 (Equation 3.26)
    modality = compute_modality_fraction(shap_array)
    with open(os.path.join(output_dir, 'modality_fraction.txt'), 'w') as f:
        f.write(f"Year: {year}\nStrategy: {strategy}\nSeed: {seed}\n"
                f"Scope: {scope}\nClasses: {class_names}\n")
        f.write(f"Patches: {len(test_idx)} / {len(dataset)}"
                f"{' (FULL)' if is_full else ''}\n\n")
        for c, cname in enumerate(class_names):
            f.write(f"  {cname}: S1_frac = {modality['s1_frac'][c]:.4f}\n")
    plot_modality_comparison(modality, class_names,
                             os.path.join(output_dir, 'modality_fraction.png'),
                             f'S1 vs S2 — {tag}')

    # 3. Cloud-conditional
    # See Subsection 3.11.2 (Equation 3.27)
    cloud_dict, n_low, n_high = compute_cloud_conditional(shap_array, test_meta)
    print(f"   Cloud: {n_low} low, {n_high} high")
    if cloud_dict:
        plot_cloud_conditional(cloud_dict, band_names,
                               os.path.join(output_dir, 'cloud_conditional.png'),
                               f'Cloud-conditional — {tag}')
        cloud_rows = []
        for key in ['low_cloud', 'high_cloud']:
            if key in cloud_dict:
                for i, bn in enumerate(band_names):
                    cloud_rows.append({'Condition': key, 'Band': bn,
                                       'Importance': cloud_dict[key].mean(axis=0)[i]})
        pd.DataFrame(cloud_rows).to_csv(
            os.path.join(output_dir, 'cloud_conditional.csv'), index=False)

    # 4. Per-facies
    # See Subsection 3.11.2
    facies_imp = compute_per_facies_importance(shap_array, test_meta)
    facies_dir = os.path.join(output_dir, 'per_facies')
    os.makedirs(facies_dir, exist_ok=True)
    for fname, imp in facies_imp.items():
        fr = {'Band': band_names}
        for c, cname in enumerate(class_names):
            fr[cname] = imp[c]
        fr['Combined'] = imp.mean(axis=0)
        pd.DataFrame(fr).sort_values('Combined', ascending=False).to_csv(
            os.path.join(facies_dir, f'{fname}.csv'), index=False)

    # 5. Spatial maps
    # See Subsection 3.11.4
    maps_dir = os.path.join(output_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)
    data_dict = load_geotiffs(get_data_root(year))
    mosaic_shap_maps(shap_array, test_meta, maps_dir,
                     data_dict['metadata'], num_classes, class_slugs)

    # 6. Raw NPZ
    np.savez_compressed(
        os.path.join(output_dir, 'shap_raw.npz'),
        shap_array=shap_array,
        importance=importance,
        band_names=np.array(band_names),
        class_names=np.array(class_names),
        s1_frac=np.array(modality['s1_frac']),
        test_indices=np.array(test_idx),
    )

    print(f"   Output: {output_dir}")
    return {
        'year': year,
        'strategy': strategy,
        'seed': seed,
        'scope': scope,
        'importance': importance,
        'modality': modality,
        'n_test_used': len(test_idx),
    }


# --- Orchestration ---
def run_all(years, strategies, seeds, scopes=('test',),
            n_background=30, n_test=-1, nsamples=50):
    """
    Iterates through all configurations to compute GradientSHAP.
    """
    configs = get_all_configs(years=years, strategies=strategies, seeds=seeds)

    print("\n" + "=" * 80)
    print("BATCH GRADIENTSHAP")
    print("=" * 80)
    print(f"Years:       {years}")
    print(f"Strategies:  {strategies}")
    print(f"Seeds:       {seeds}")
    print(f"Scopes:      {list(scopes)}")
    print(f"Total runs:  {len(configs) * len(scopes)}")

    results = []
    for cfg in configs:
        for scope in scopes:
            try:
                r = run_gradient_shap(
                    cfg['year'], cfg['split_type'], cfg['seed'],
                    scope=scope,
                    n_background=n_background,
                    n_test=n_test,
                    nsamples=nsamples,
                )
                results.append(r)
            except Exception as e:
                print(f"\n  FAILED: {cfg['name']} [{scope}]: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Successful: {len(results)} / {len(configs) * len(scopes)}")
    return results


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute GradientSHAP feature attributions."
    )
    parser.add_argument("--year", type=str, default="both",
                        choices=["2021", "2022", "both"])
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["random", "blocking", "all"])
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seeds, or 'all'.")
    parser.add_argument("--scope", type=str, default="test",
                        choices=["test", "all", "both"])
    parser.add_argument("--n_background", type=int, default=30)
    parser.add_argument("--n_test", type=int, default=-1,
                        help="-1 = use FULL set in scope")
    parser.add_argument("--nsamples", type=int, default=50)
    args = parser.parse_args()

    years = resolve_years(args.year)
    strategies = resolve_strategies(args.strategy)
    seeds = resolve_seeds(args.seeds)
    scopes = ("test", "all") if args.scope == "both" else (args.scope,)

    run_all(years, strategies, seeds, scopes,
            n_background=args.n_background,
            n_test=args.n_test,
            nsamples=args.nsamples)