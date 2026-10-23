"""
inference.py

Full AOI prediction, probability generation, and Gaussian mosaicking.
Reference: Subsections 3.9.1, 3.9.2, and 3.9.3.
"""

import os
import csv
import argparse
import numpy as np
import torch
import rasterio
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from .config import (
    get_config,
    get_all_configs,
    resolve_years,
    resolve_strategies,
    resolve_seeds,
    get_checkpoint_path,
    get_inference_dir,
    get_class_slugs,
    get_model_dir,
)
from .models import get_model
from .data_loader import load_geotiffs
from .metrics import compute_patch_iou
from .xai_utils import create_gaussian_weight


# --- Helper Functions ---
def get_patch_starts(total_size, patch_size, stride):
    """
    Generates sliding window start coordinates for inference.
    """
    # See Subsection 3.9.1 (Full-AOI Inference Gaussian Mosaicking)
    starts = list(range(0, total_size - patch_size + 1, stride))
    if starts[-1] != total_size - patch_size:
        starts.append(total_size - patch_size)
    return starts


def apply_cloud_mask_to_s2(image_patch, cloud_patch):
    """
    Zero out Sentinel-2 bands (positions 4-13) for cloudy pixels.
    """
    # See Subsection 3.4.3 (Cloud Masking with SCL Band)
    for pos in range(4, 14):
        if pos < image_patch.shape[2]:
            image_patch[:, :, pos] = image_patch[:, :, pos] * cloud_patch
    return image_patch


def run_inference_on_patch(model, patch, device):
    """
    Executes a forward pass on a single patch and returns softmax probabilities.
    """
    # See Subsection 3.9.1
    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(patch_tensor)
        probs = F.softmax(logits, dim=1)
    return probs.squeeze(0).cpu().numpy()


def mosaic_patches_gaussian(probs_stack, starts_y, starts_x, patch_size,
                            H, W, num_classes, sigma=64):
    """
    Combines patch predictions into a full-AOI mosaic using Gaussian weighting.
    """
    # See Subsection 3.9.1 (Full-AOI Inference Gaussian Mosaicking)
    weight_map = create_gaussian_weight(patch_size, sigma)
    mosaic = np.zeros((num_classes, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    idx = 0
    for y in starts_y:
        for x in starts_x:
            # Apply Gaussian weight
            # See Equation 3.12 (Gaussian Weight)
            mosaic[:, y:y+patch_size, x:x+patch_size] += probs_stack[idx] * weight_map
            wsum[y:y+patch_size, x:x+patch_size] += weight_map
            idx += 1

    wsum = np.maximum(wsum, 1e-8)
    
    # Calculate weighted average probabilities
    # See Equation 3.13 (Weighted Probability Mosaicking)
    return mosaic / wsum, wsum


def load_assignment_map(year, strategy, seed):
    """
    Reads patch assignment CSVs to map coordinates to their split values.
    """
    # See Subsection 3.10.2 (Train-Test Spatial Proximity)
    model_dir = get_model_dir(year, strategy, seed)
    assignment_map = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(model_dir, f"{split_name}_patch_assignments.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                yy = int(float(row[0]))
                xx = int(float(row[1]))
                sv = int(float(row[2]))
                assignment_map[(yy, xx)] = sv
    return assignment_map


# --- Visualization Functions ---
def plot_patch_assignment(patch_results, H, W, patch_size, output_path, title):
    """
    Visualizes the spatial layout of train/val/test patches.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    has_split = patch_results and 'split_value' in patch_results[0]

    if has_split:
        split_map = np.full((H, W), np.nan, dtype=np.float32)
        for r in patch_results:
            if r['split_value'] > 0: 
                split_map[r['y']:r['y']+patch_size,
                          r['x']:r['x']+patch_size] = r['split_value']
                
        cmap = ListedColormap(['#2ca02c', '#ff7f0e', '#1f77b4'])
        cmap.set_bad(alpha=0.0)
        
        ax.imshow(split_map, cmap=cmap, alpha=0.7, vmin=1, vmax=3)
        
        patches = [
            mpatches.Patch(color='#2ca02c', label='Train'),
            mpatches.Patch(color='#ff7f0e', label='Validation'),
            mpatches.Patch(color='#1f77b4', label='Test'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=10)

    for r in patch_results:
        if not has_split or r.get('split_value', 0) > 0:
            rect = plt.Rectangle((r['x'], r['y']), patch_size, patch_size,
                                 fill=False, edgecolor='white',
                                 linewidth=0.5, alpha=0.5)
            ax.add_patch(rect)

    for r in patch_results:
        rect = plt.Rectangle((r['x'], r['y']), patch_size, patch_size,
                             fill=False, edgecolor='white',
                             linewidth=0.5, alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=14); ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_patch_quality_heatmap(patch_results, H, W, patch_size,
                               output_path, title):
    """
    Visualizes the spatial distribution of per-patch IoU scores.
    """
    quality = np.full((H, W), np.nan, dtype=np.float32)
    for r in patch_results:
        if 'iou' in r:
            quality[r['y']:r['y']+patch_size,
                    r['x']:r['x']+patch_size] = r['iou']

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    im = ax.imshow(quality, cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('IoU', fontsize=12)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=14); ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Main Inference Routine ---
def run_inference(config, visualize_patches=True):
    """
    Executes full-AOI prediction and mosaicking for a single model configuration.
    """
    year = config['year']
    strategy = config['split_type']
    seed = config['seed']
    num_classes = config['num_classes']
    class_slugs = get_class_slugs(year)

    tag = f"{year}_{strategy}_seed{seed}"
    print("\n" + "=" * 80)
    print(f"RUNNING INFERENCE: {tag}")
    print("=" * 80)

    # Verify Checkpoint
    ckpt_path = get_checkpoint_path(year, strategy, seed)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_root = config['data_root']
    output_dir = get_inference_dir(year, strategy, seed)
    os.makedirs(output_dir, exist_ok=True)

    device = config['device']
    patch_size = config['patch_size']
    stride = config['stride']

    # Load Model
    print("\nLoading model...")
    model = get_model(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"   Checkpoint:     {ckpt_path}")
    print(f"   Trained epoch:  {ckpt.get('epoch', 'N/A')}")
    print(f"   Val loss:       {ckpt.get('val_loss', 'N/A')}")

    # Load AOI Data
    print(f"\nLoading {year} AOI...")
    data_dict = load_geotiffs(data_root)
    image_stack = data_dict['image_stack']
    cloud_mask = data_dict['cloud_mask']
    metadata = data_dict['metadata']
    H, W, C = image_stack.shape
    print(f"   Image shape: {H} x {W} x {C}")

    starts_y = get_patch_starts(H, patch_size, stride)
    starts_x = get_patch_starts(W, patch_size, stride)
    total_patches = len(starts_y) * len(starts_x)
    print(f"   Patch grid: {len(starts_y)} x {len(starts_x)} = {total_patches}")

    # Sliding Window Inference
    # See Subsection 3.9.1
    print("\nRunning inference...")
    probs_stack = []
    patch_results = []
    with tqdm(total=total_patches, desc="   Patches") as pbar:
        for y in starts_y:
            for x in starts_x:
                patch = image_stack[y:y+patch_size, x:x+patch_size, :].copy()
                cloud = cloud_mask[y:y+patch_size, x:x+patch_size]
                patch = apply_cloud_mask_to_s2(patch, cloud)
                patch_t = np.transpose(patch, (2, 0, 1)).astype(np.float32)
                probs = run_inference_on_patch(model, patch_t, device)
                probs_stack.append(probs)
                patch_results.append({
                    'y': y, 'x': x, 'pred': np.argmax(probs, axis=0)
                })
                pbar.update(1)

    # Patch-level IoU and Split Assignment
    # See Subsection 3.9.3 (Evaluation Scope and Metrics)
    assignment_map = load_assignment_map(year, strategy, seed)
    try:
        label = data_dict['label']
        for r in patch_results:
            y, x = r['y'], r['x']
            label_patch = label[y:y+patch_size, x:x+patch_size]
            r['iou'] = compute_patch_iou(
                r['pred'], label_patch, num_classes=num_classes
            )
            r['split_value'] = assignment_map.get((y, x), 0)
    except Exception as e:
        print(f"   Could not compute patch-level IoU: {e}")

    # Mosaic Predictions
    print("\nMosaicking...")
    mosaic_probs, weight_sum = mosaic_patches_gaussian(
        probs_stack, starts_y, starts_x, patch_size, H, W,
        num_classes=num_classes, sigma=64,
    )

    class_map = np.argmax(mosaic_probs, axis=0).astype(np.uint8)

    # Class Distribution Statistics
    class_counts = [(class_map == c).sum() for c in range(num_classes)]
    total_px = H * W
    print(f"\n   Class distribution:")
    for c, name in enumerate(config['class_names']):
        n = int(class_counts[c])
        print(f"     {name:<12} {n:>10,}  ({n/total_px*100:5.2f}%)")

    # Save GeoTIFF Outputs
    crs = metadata['crs']
    transform = metadata['transform']

    with rasterio.open(os.path.join(output_dir, "classification_map.tif"), 'w',
                       driver='GTiff', height=H, width=W, count=1,
                       dtype='uint8', crs=crs, transform=transform,
                       compress='lzw') as dst:
        dst.write(class_map, 1)

    for i, slug in enumerate(class_slugs):
        with rasterio.open(os.path.join(output_dir, f"prob_{slug}.tif"), 'w',
                           driver='GTiff', height=H, width=W, count=1,
                           dtype='float32', crs=crs, transform=transform,
                           compress='lzw') as dst:
            dst.write(mosaic_probs[i], 1)

    with rasterio.open(os.path.join(output_dir, "weight_sum.tif"), 'w',
                       driver='GTiff', height=H, width=W, count=1,
                       dtype='float32', crs=crs, transform=transform,
                       compress='lzw') as dst:
        dst.write(weight_sum, 1)

    print(f"   GeoTIFFs saved to: {output_dir}")

    # Save Patch Visualizations
    if visualize_patches and patch_results:
        vis_dir = os.path.join(output_dir, "patch_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        plot_patch_assignment(
            patch_results, H, W, patch_size,
            os.path.join(vis_dir, "patch_assignment.png"),
            f"Patch Assignment - {tag}"
        )
        if 'iou' in patch_results[0]:
            plot_patch_quality_heatmap(
                patch_results, H, W, patch_size,
                os.path.join(vis_dir, "patch_quality_heatmap.png"),
                f"Patch Quality - {tag}"
            )

    # Save Text Summary
    with open(os.path.join(output_dir, "inference_summary.txt"), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"INFERENCE SUMMARY: {tag}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Year:          {year}\n")
        f.write(f"Strategy:      {strategy}\n")
        f.write(f"Seed:          {seed}\n")
        f.write(f"Num classes:   {num_classes}\n")
        f.write(f"Checkpoint:    {ckpt_path}\n\n")
        f.write(f"Image dimensions: {H} x {W}\n")
        f.write(f"Patch size:       {patch_size}\n")
        f.write(f"Stride:           {stride}\n")
        f.write(f"Total patches:    {total_patches}\n")
        f.write(f"Mosaicking:       Gaussian weighting (sigma=64)\n\n")
        f.write("Class distribution:\n")
        for c, name in enumerate(config['class_names']):
            n = int(class_counts[c])
            f.write(f"  {name:<12} {n:>10,}  ({n/total_px*100:5.2f}%)\n")

        if patch_results and 'iou' in patch_results[0]:
            ious = [r['iou'] for r in patch_results if 'iou' in r]
            if ious:
                f.write(f"\nPatch-level IoU statistics:\n")
                f.write(f"  Mean IoU: {np.mean(ious):.4f}\n")
                f.write(f"  Std IoU:  {np.std(ious):.4f}\n")
                f.write(f"  Min IoU:  {np.min(ious):.4f}\n")
                f.write(f"  Max IoU:  {np.max(ious):.4f}\n")

    print(f"   Summary: {os.path.join(output_dir, 'inference_summary.txt')}")

    return {
        'name': tag,
        'year': year,
        'strategy': strategy,
        'seed': seed,
        'output_dir': output_dir,
        'class_counts': class_counts,
    }


# --- Orchestrator ---
def run_all_inference(configs, visualize_patches=True):
    """
    Iterates through all configurations and runs the inference pipeline.
    """
    print("\n" + "=" * 80)
    print("BATCH INFERENCE")
    print("=" * 80)
    print(f"Total runs: {len(configs)}")

    results = []
    for cfg in configs:
        try:
            r = run_inference(cfg, visualize_patches=visualize_patches)
            results.append(r)
        except FileNotFoundError as e:
            print(f"\n  Skipping {cfg['name']}: {e}")
        except Exception as e:
            print(f"\n  ERROR on {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("BATCH INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Successful: {len(results)} / {len(configs)}")
    for r in results:
        counts = r['class_counts']
        total = sum(counts)
        pcts = "  ".join(f"{c/total*100:5.2f}%" for c in counts)
        print(f"  {r['name']:<30}  {pcts}")

    return results


# --- CLI Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for multiple years, strategies, and seeds."
    )
    parser.add_argument("--year", type=str, default="both",
                        choices=["2021", "2022", "both"])
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["random", "blocking", "all"])
    parser.add_argument("--seeds", type=str, default="all",
                        help="Comma-separated seeds, or 'all'.")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable patch visualizations.")
    args = parser.parse_args()

    years = resolve_years(args.year)
    strategies = resolve_strategies(args.strategy)
    seeds = resolve_seeds(args.seeds)

    configs = get_all_configs(years=years, strategies=strategies, seeds=seeds)

    run_all_inference(configs, visualize_patches=not args.no_vis)