"""
xai_utils.py

Shared utilities for XAI analysis and spatial probability mapping.
Reference: Subsections 3.7.1, 3.9.1, and 3.11.4.
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path


# --- Gaussian Weighting ---
def create_gaussian_weight(patch_size, sigma=64):
    """
    Generates a 2D Gaussian weight map for seamless patch mosaicking.
    """
    # See Subsection 3.9.1 (Full-AOI Inference Gaussian Mosaicking)
    # See Equation 3.12 (Gaussian Weight)
    x = np.arange(patch_size)
    y = np.arange(patch_size)
    xx, yy = np.meshgrid(x, y)
    center = patch_size // 2
    weight = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * sigma**2))
    return weight / weight.max()


# --- GeoTIFF Export ---
def save_geotiff(data, output_path, reference_meta, dtype='float32'):
    """
    Saves a 2D numpy array as a GeoTIFF using spatial metadata from a reference image.
    """
    # See Subsection 3.11.4 (Spatial Attribution Map)
    H, W = data.shape[-2:]

    profile = {
        'driver': 'GTiff',
        'height': H,
        'width': W,
        'count': 1,
        'dtype': dtype,
        'crs': reference_meta.get('crs'),
        'transform': reference_meta.get('transform'),
        'compress': 'lzw',
    }

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

    return output_path


# --- Band Naming ---
def get_band_names(config=None):
    """
    Returns the canonical 14-band naming scheme for SHAP analysis.
    """
    # See Subsection 3.7.1 (Dual Encoder Architecture)
    # Always returns 14 bands: 4 Sentinel-1 + 10 Sentinel-2
    band_names = [
        'S1_VV_Desc',
        'S1_VH_Desc',
        'S1_VV_Asc',
        'S1_VH_Asc',
    ]
    band_names.extend([
        'S2_B2 (Blue)',
        'S2_B3 (Green)',
        'S2_B4 (Red)',
        'S2_B5 (RE1)',
        'S2_B6 (RE2)',
        'S2_B7 (RE3)',
        'S2_B8 (NIR)',
        'S2_B8A (Narrow NIR)',
        'S2_B11 (SWIR1)',
        'S2_B12 (SWIR2)',
    ])
    return band_names