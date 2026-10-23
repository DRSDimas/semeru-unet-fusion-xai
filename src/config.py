"""
config.py

Global configuration and experiment registry for the Gated U-Net framework.
Reference: Subsections 3.4.4, 3.5, 3.6.2, 3.6.3, 3.7.1, and 3.8.
"""

import os
import torch

# --- Band Index Mapping ---
# See Subsection 3.7.1 (Dual Encoder Architecture)
# Tensor input has 14 channels: 4 Sentinel-1 (backscatter) + 10 Sentinel-2 (reflectance).
BAND_INFO = {
    # Sentinel-1 (Channels 0-3)
    "S1_desc_VV": 0,
    "S1_desc_VH": 1,
    "S1_asc_VV": 2,
    "S1_asc_VH": 3,
    # Sentinel-2 (Channels 4-13)
    "S2_B2": 4,
    "S2_B3": 5,
    "S2_B4": 6,
    "S2_B5": 7,
    "S2_B6": 8,
    "S2_B7": 9,
    "S2_B8": 10,
    "S2_B8A": 11,
    "S2_B11": 12,
    "S2_B12": 13,
}

# --- Class Naming ---
# See Subsection 3.4.4 (UNOSAT Reference Labels)
# 2021 utilizes the 3-class UNOSAT scheme.
CLASS_NAMES_2021 = ["Background", "PDC", "PDC+Lahar"]
CLASS_SLUGS_2021 = ["bg", "pdc", "pdc_lahar"]

# 2022 utilizes a binary scheme, as UNOSAT does not distinguish PDC from lahar.
CLASS_NAMES_2022 = ["Background", "Deposit"]
CLASS_SLUGS_2022 = ["bg", "deposit"]

def get_class_names(year):
    return CLASS_NAMES_2021 if str(year) == "2021" else CLASS_NAMES_2022

def get_class_slugs(year):
    return CLASS_SLUGS_2021 if str(year) == "2021" else CLASS_SLUGS_2022

# --- Facies Mapping ---
# See Subsections 3.5 (Facies Delineation) & 3.6.2 (Blocking Split)
# Raster values: 1=Central, 2=Proximal, 3=Medial, 4=Distal, 5=Inlier
FACIES_NAMES = ["Central", "Proximal", "Medial", "Distal", "Inlier"]
FACIES_COLORS = ["#e41a1c", "#de820a", "#b39223", "#e8e50d", "#ff7f00"]

# Facies-to-subset assignment for the blocking strategy. 
# The Medial facies (3) acts as a spatial buffer between Train and Test.
FACIES_SPLIT_MAP = {
    1: 1,  # Central   -> Training
    2: 1,  # Proximal  -> Training
    3: 2,  # Medial    -> Validation (Buffer)
    4: 3,  # Distal    -> Test
    5: 3,  # Inlier    -> Test
}

# --- Experiment Registry ---
# See Subsection 3.8.3 (Training Budget & Reproducibility)
SEEDS = [42, 123, 456, 789, 999]
STRATEGIES = ["random", "blocking"]
YEARS = ["2021", "2022"]

# Relative paths for compatibility
EXPERIMENTS_ROOT = "./experiments/"
DATA_ROOT_BASE = "./data/raw"

# --- Base Configuration ---
BASE_CONFIG = {
    "experiments_root": EXPERIMENTS_ROOT,

    # See Subsection 3.6.1 (Patch Extraction)
    "patch_size": 160,
    "stride": 80,
    
    # See Subsection 3.8.2 & Table 3.6 (Hyperparameters)
    "batch_size": 16,
    "num_workers": 0,
    "epochs": 100,           
    "learning_rate": 1e-4,

    # See Subsection 3.7.1 (Architecture)
    "encoder_name": "resnet34",
    "encoder_weights": None,
    "num_classes": 3,         # Overridden per year via get_config()
    "s1_channels": 4,
    "s2_channels": 10,

    # See Subsection 3.8.1 & Equation 3.10 (Focal Loss)
    "focal_gamma": 2.0,
    "class_weights": None,    # Recomputed dynamically from the training split

    # See Subsection 3.7.1 (Channel splitting)
    "s1_band_indices": list(range(4)),
    "s2_band_indices": list(range(4, 14)),

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# --- Path Helpers ---
def get_data_root(year):
    """Path to year-specific raw data folder."""
    return os.path.join(DATA_ROOT_BASE, str(year))

def get_model_name(year, strategy, seed):
    """Canonical model identifier. Example: '2021_random_seed42'."""
    return f"{year}_{strategy}_seed{seed}"

def get_model_dir(year, strategy, seed):
    """Top-level output folder for a trained model."""
    return os.path.join(EXPERIMENTS_ROOT, get_model_name(year, strategy, seed))

def get_checkpoint_path(year, strategy, seed):
    return os.path.join(get_model_dir(year, strategy, seed), "checkpoints", "best_model.pth")

def get_summary_path(year, strategy, seed):
    return os.path.join(get_model_dir(year, strategy, seed), "summary.txt")

def get_history_path(year, strategy, seed):
    return os.path.join(get_model_dir(year, strategy, seed), "logs", "training_history.csv")

def get_inference_dir(year, strategy, seed):
    return os.path.join(get_model_dir(year, strategy, seed), "predictions")

def get_shap_dir(year, strategy, seed, scope="test"):
    """SHAP outputs for a model, scoped to its own training year."""
    return os.path.join(get_model_dir(year, strategy, seed), "shap_gradient", scope)


# --- Config Helpers ---
def get_config(year, strategy, seed):
    """Build a full config dict for a (year, strategy, seed) triplet."""
    year = str(year)
    if year not in YEARS:
        raise ValueError(f"Unknown year '{year}'. Choose from {YEARS}.")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed {seed}. Choose from {SEEDS}.")

    cfg = BASE_CONFIG.copy()
    cfg.update({
        "name": get_model_name(year, strategy, seed),
        "year": year,
        "seed": seed,
        "split_type": strategy,
        "data_root": get_data_root(year),
        "model_dir": get_model_dir(year, strategy, seed),
        "num_classes": 3 if year == "2021" else 2,
        "class_names": get_class_names(year),
        "class_slugs": get_class_slugs(year),
    })
    return cfg

def get_all_configs(years=None, strategies=None, seeds=None):
    """Build configs for the cross-product of (year, strategy, seed)."""
    if years is None or years == "both":
        years = YEARS
    if strategies is None or strategies == "all":
        strategies = STRATEGIES
    if seeds is None or seeds == "all":
        seeds = SEEDS

    return [get_config(y, s, seed) for y in years for s in strategies for seed in seeds]


# --- CLI Argument Resolution ---
def resolve_seeds(seeds_arg):
    if seeds_arg is None or seeds_arg == "all":
        return list(SEEDS)
    if isinstance(seeds_arg, str):
        parts = [p.strip() for p in seeds_arg.split(",") if p.strip()]
        return [int(p) for p in parts]
    return list(seeds_arg)

def resolve_strategies(strategy_arg):
    if strategy_arg is None or strategy_arg == "all":
        return list(STRATEGIES)
    if strategy_arg not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_arg}'. Choose from {STRATEGIES} or 'all'.")
    return [strategy_arg]

def resolve_years(year_arg):
    if year_arg is None or year_arg == "both":
        return list(YEARS)
    year_arg = str(year_arg)
    if year_arg not in YEARS:
        raise ValueError(f"Unknown year '{year_arg}'. Choose from {YEARS} or 'both'.")
    return [year_arg]

# --- Strategy Descriptions ---
# See Subsections 3.6.2 & 3.6.3
STRATEGY_DESCRIPTIONS = {
    "random": "Random split (47.4/26.5/26.1%). Preserves the spatial distribution of facies across all three subsets without spatial constraints.",
    "blocking": "Facies-based blocking split. Central+Proximal (Train), Medial (Validation), Distal+Inlier (Test). The Medial facies acts as a spatial buffer to prevent leakage."
}