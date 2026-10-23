"""
data_loader.py

Data loading and patch extraction for the Semeru eruption materials mapping.
Reference: Subsections 3.4.3, 3.4.4, 3.6.1, 3.6.2, 3.6.3, 3.7.1, and 3.10.2.
"""

import os
import csv
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


# --- GeoTIFF Loading ---
def load_geotiffs(data_root):
    """
    Loads year-specific GeoTIFFs and timeless facies data.
    """
    data_root_norm = os.path.normpath(data_root)
    parent_root = os.path.dirname(data_root_norm)

    year_files = {
        "s1":    "S1_Backscatter_Norm.tif",
        "s2":    "S2_Reflectance_Norm.tif",
        "label": "Label.tif",
        "cloud": "S2_CloudMask.tif",
    }
    timeless_files = {
        "split":  "Facies_Split.tif",
        "facies": "Facies_Class.tif",
    }

    data = {}
    metadata = {}

    for key, filename in year_files.items():
        path = os.path.join(data_root_norm, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Year-specific file not found: {path}. "
                f"Expected {filename} in {data_root_norm}"
            )
        with rasterio.open(path) as src:
            data[key] = src.read() if key in ("s1", "s2") else src.read(1)
            if key == "label":
                metadata["crs"] = src.crs
                metadata["transform"] = src.transform
                metadata["width"] = src.width
                metadata["height"] = src.height

    for key, filename in timeless_files.items():
        path = os.path.join(parent_root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Timeless file not found: {path}. "
                f"Expected {filename} in {parent_root}"
            )
        with rasterio.open(path) as src:
            data[key] = src.read(1)

    # Stack S1 (4 bands) and S2 (10 bands) into a 14-channel tensor
    # See Subsection 3.7.1 (Dual Encoder Architecture)
    image_stack = np.concatenate([data["s1"], data["s2"]], axis=0)
    image_stack = np.transpose(image_stack, (1, 2, 0))  # Shape: (H, W, 14)

    # Remap raw labels: 1->0, 2->1, 3->2 (and 255->254 for nodata)
    # See Subsection 3.4.4 (UNOSAT Reference Labels)
    label = data["label"] - 1

    # Binary cloud mask: 1=clear, 0=cloudy
    # See Subsection 3.4.3 (Cloud Masking with SCL Band)
    cloud_binary = np.ones_like(data["cloud"], dtype=np.float32)
    cloud_binary[data["cloud"] == 2] = 0

    print("   Loaded data:")
    print(f"   Image stack:  {image_stack.shape} (H, W, C)")
    print(f"   Label:        {label.shape}  "
          f"(unique values: {sorted(np.unique(label).tolist())[:8]}...)")
    print(f"   Split mask:   {data['split'].shape}")
    print(f"   Cloud mask:   {cloud_binary.shape}")
    print(f"   Facies mask:  {data['facies'].shape}")

    return {
        "image_stack": image_stack,
        "label": label,
        "split_mask": data["split"],
        "cloud_mask": cloud_binary,
        "facies": data["facies"],
        "metadata": metadata,
    }


# --- Patch Dataset ---
class SemeruPatchDataset(Dataset):
    """
    Sliding-window patch dataset for model training, validation, and testing.
    """

    VALID_SPLIT_TYPES = ("train", "val", "test", "all")
    SPLIT_VALUES = {"train": 1, "val": 2, "test": 3}

    def __init__(self, data_dict, split_type="train", patch_size=160,
                 stride=80, augment=False, split_strategy="random",
                 random_seed=42):
        if split_type not in self.VALID_SPLIT_TYPES:
            raise ValueError(
                f"Unknown split_type '{split_type}'. "
                f"Choose from {self.VALID_SPLIT_TYPES}."
            )
        if split_strategy not in ("random", "blocking"):
            raise ValueError(
                f"Unknown split_strategy '{split_strategy}'. "
                f"Choose 'random' or 'blocking'."
            )

        self.data = data_dict
        self.split_type = split_type
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.split_strategy = split_strategy
        self.random_seed = random_seed

        self.image_stack = data_dict["image_stack"]
        self.label = data_dict["label"]
        self.split_mask = data_dict["split_mask"]
        self.cloud_mask = data_dict["cloud_mask"]
        self.facies = data_dict["facies"]

        self.patch_coords = self._generate_patches()

        print(f"    [{split_strategy:>8}] {split_type:>5}: "
              f"{len(self.patch_coords):>3} patches")

    def _generate_patches(self):
        """
        Generates patch coordinates using a sliding window.
        """
        H, W = self.label.shape
        all_patches = []
        
        # Generate Y starts
        # See Subsection 3.6.1 (Patch Extraction)
        starts_y = list(range(0, H - self.patch_size + 1, self.stride))
        if starts_y[-1] != H - self.patch_size:
            starts_y.append(H - self.patch_size)
            
        # Generate X starts
        # See Subsection 3.6.1 (Patch Extraction)
        starts_x = list(range(0, W - self.patch_size + 1, self.stride))
        if starts_x[-1] != W - self.patch_size:
            starts_x.append(W - self.patch_size)

        for y in starts_y:
            for x in starts_x:
                all_patches.append((y, x))

        self._patch_assignments = []

        # Dispatch to the selected split strategy
        if self.split_strategy == "random":
            return self._generate_random(all_patches)
        elif self.split_strategy == "blocking":
            return self._generate_blocking(all_patches)
        
        return all_patches

    def _generate_random(self, all_patches):
        """
        Assigns patches to subsets using a random split.
        """
        # See Subsection 3.6.3 (Random Split)
        rng = np.random.RandomState(self.random_seed)
        shuffled = rng.permutation(len(all_patches))
        
        # Proportions 47.4% / 26.5% / 26.1%
        n_train = int(len(all_patches) * 0.474)
        n_val = int(len(all_patches) * 0.265)

        if self.split_type == "all":
            self._patch_assignments = [
                (y, x, self._majority_split(y, x)) for (y, x) in all_patches
            ]
            return list(all_patches)

        if self.split_type == "train":
            idx = shuffled[:n_train]
        elif self.split_type == "val":
            idx = shuffled[n_train:n_train + n_val]
        else:
            idx = shuffled[n_train + n_val:]

        patches = [all_patches[i] for i in idx]
        sv = self.SPLIT_VALUES[self.split_type]
        self._patch_assignments = [(y, x, sv) for (y, x) in patches]
        return patches

    def _generate_blocking(self, all_patches):
        """
        Assigns patches to subsets using facies-based spatial blocking.
        """
        # See Subsection 3.6.2 (Facies-Based Blocking Split)
        if self.split_type == "all":
            self._patch_assignments = [
                (y, x, self._majority_split(y, x)) for (y, x) in all_patches
            ]
            return list(all_patches)

        target = self.SPLIT_VALUES[self.split_type]
        patches = []
        for (y, x) in all_patches:
            sv = self._majority_split(y, x)
            if sv == target:
                patches.append((y, x))
                self._patch_assignments.append((y, x, sv))
        return patches

    def _majority_split(self, y, x):
        """
        Determines the split assignment of a patch via majority vote.
        """
        # See Subsection 3.6.2 (Facies-Based Blocking Split)
        facies_patch = self.facies[y:y+self.patch_size, x:x+self.patch_size]
        u_f, c_f = np.unique(facies_patch, return_counts=True)
        if len(u_f) > 0:
            majority_facies = u_f[np.argmax(c_f)]
            if majority_facies == 0: 
                return 0

        patch = self.split_mask[y:y+self.patch_size, x:x+self.patch_size]
        u, c = np.unique(patch, return_counts=True)
        return int(u[np.argmax(c)]) if len(u) > 0 else 0

    def _apply_augmentation(self, image, label, cloud):
        """
        Applies geometric augmentation (flipping and rotation).
        """
        # See Subsection 3.6.1 (Patch Extraction)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
            cloud = np.flip(cloud, axis=1)
            
        if np.random.random() > 0.5:
            image = np.flip(image, axis=0)
            label = np.flip(label, axis=0)
            cloud = np.flip(cloud, axis=0)
            
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1))
            label = np.rot90(label, k, axes=(0, 1))
            cloud = np.rot90(cloud, k, axes=(0, 1))
            
        return image, label, cloud

    def save_patch_assignments(self, csv_path):
        """
        Saves patch coordinates and split assignments to a CSV file.
        """
        # See Subsection 3.10.2 (Train-Test Spatial Proximity)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["patch_id", "y_start", "x_start", "split_value"])
            for idx, (y, x, sv) in enumerate(self._patch_assignments):
                writer.writerow([idx, y, x, sv])

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, idx):
        y, x = self.patch_coords[idx]
        ps = self.patch_size

        # Copy to avoid mutating the underlying stack when cloud-masking
        image = self.image_stack[y:y+ps, x:x+ps, :].copy()
        cloud = self.cloud_mask[y:y+ps, x:x+ps]

        # Zero out S2 bands under clouds to force implicit modality reliance
        # See Subsection 3.7.2 (Gating Mechanism)
        for pos in range(4, 14):
            if pos < image.shape[2]:
                image[:, :, pos] *= cloud

        label = self.label[y:y+ps, x:x+ps]
        facies = self.facies[y:y+ps, x:x+ps]

        if self.augment and self.split_type == "train":
            image, label, cloud = self._apply_augmentation(image, label, cloud)

        # Convert to PyTorch tensors: (C, H, W) for images
        image_t = torch.from_numpy(
            np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        ).float()
        label_t = torch.from_numpy(np.ascontiguousarray(label)).long()
        cloud_t = torch.from_numpy(np.ascontiguousarray(cloud)).float()
        facies_t = torch.from_numpy(np.ascontiguousarray(facies)).long()

        return image_t, label_t, cloud_t, facies_t


# --- DataLoader Factory ---
def create_dataloaders(config):
    """
    Builds train, validation, and test DataLoaders for a single model configuration.
    """
    data_dict = load_geotiffs(config["data_root"])
    split_strategy = config["split_type"]
    patch_size = config["patch_size"]
    stride = config["stride"]

    common = dict(
        data_dict=data_dict,
        patch_size=patch_size,
        stride=stride,
        split_strategy=split_strategy,
        random_seed=42,
    )

    train_ds = SemeruPatchDataset(split_type="train", augment=True, **common)
    val_ds = SemeruPatchDataset(split_type="val", augment=False, **common)
    test_ds = SemeruPatchDataset(split_type="test", augment=False, **common)

    n_tr, n_v, n_te = len(train_ds), len(val_ds), len(test_ds)
    n_tot = n_tr + n_v + n_te
    if n_tot > 0:
        print(f"\n   Patch-level split ({split_strategy}):")
        print(f"     Train: {n_tr:>3}  ({n_tr/n_tot*100:.1f}%)")
        print(f"     Val:   {n_v:>3}  ({n_v/n_tot*100:.1f}%)")
        print(f"     Test:  {n_te:>3}  ({n_te/n_tot*100:.1f}%)")

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # Save patch assignments for downstream structural diagnostics
    # See Subsection 3.10.2 (Train-Test Spatial Proximity)
    model_dir = config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    train_ds.save_patch_assignments(
        os.path.join(model_dir, "train_patch_assignments.csv"))
    val_ds.save_patch_assignments(
        os.path.join(model_dir, "val_patch_assignments.csv"))
    test_ds.save_patch_assignments(
        os.path.join(model_dir, "test_patch_assignments.csv"))

    return train_loader, val_loader, test_loader, data_dict