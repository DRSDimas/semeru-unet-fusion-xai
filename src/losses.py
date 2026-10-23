"""
losses.py

Loss function definitions for the Semeru eruption deposit mapping framework.
Reference: Subsection 3.8.1 (Loss Function and Class Weighting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Cloud Mask Policy ---
# See Subsection 3.8.1. Sentinel-2 bands are zeroed out where cloud_mask == 0 in the data loader. The loss is computed on all pixels, including cloudy ones. 
# This forces the model to implicitly learn to rely on Sentinel-1 when Sentinel-2 is missing. The cloud mask is not used to weight or ignore pixels in the loss.

# --- Focal Loss ---
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation to address class imbalance.
    See Equation 3.10.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cuda'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
            else:
                self.alpha = alpha.to(device)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, C, H, W) model output (raw logits)
            targets: (B, H, W) class labels in [0, C-1]
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Extract probability of the true class (p_t)
        pt = (probs * targets_one_hot).sum(dim=1)

        # Compute focal weight: (1 - p_t)^gamma
        # See Equation 3.10
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute standard cross-entropy: -log(p_t)
        ce_loss = -torch.log(pt + 1e-8)

        loss_per_pixel = focal_weight * ce_loss

        # Apply class weights (alpha_t) if provided
        # See Equation 3.11 (Inverse Square Root Weighting)
        if self.alpha is not None:
            alpha_expanded = self.alpha.view(1, -1, 1, 1)
            alpha_t = (targets_one_hot * alpha_expanded).sum(dim=1)
            loss_per_pixel = loss_per_pixel * alpha_t

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss_per_pixel.mean()
        elif self.reduction == 'sum':
            loss = loss_per_pixel.sum()
        else:
            loss = loss_per_pixel

        return loss


# --- Loss Factory ---
def get_loss_function(config, device='cuda'):
    """
    Factory function to instantiate the Focal Loss with the specified configuration.
    """
    # See Subsection 3.8.1 (Loss Function and Class Weighting)
    class_weights = config.get("class_weights", None)
    focal_gamma = config.get("focal_gamma", 2.0)

    print(f"   Using Focal Loss (gamma={focal_gamma})")
    if class_weights is not None:
        print(f"   Class weights: {class_weights}")

    loss_fn = FocalLoss(
        alpha=class_weights,
        gamma=focal_gamma,
        device=device,
    )

    return loss_fn