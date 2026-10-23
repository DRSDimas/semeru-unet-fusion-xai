"""
models.py

Gated U-Net architecture for multisensor fusion.
Reference: Subsections 3.7.1, 3.7.2, and 3.7.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


# --- Gating Module ---
class GatingModule(nn.Module):
    """
    Multiplicative gating module for adaptive fusion.
    See Subsection 3.7.2 (Gating Mechanism).
    """
    def __init__(self, feature_dim):
        super(GatingModule, self).__init__()
        self.feature_dim = feature_dim
        
        # Define gate for Sentinel-1
        # See Equation 3.7
        self.gate_s1 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Define gate for Sentinel-2
        # See Equation 3.7
        self.gate_s2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, s1_features, s2_features):
        weight_s1 = self.gate_s1(s1_features)
        weight_s2 = self.gate_s2(s2_features)
        
        # Element-wise weighted fusion
        # See Equation 3.8
        fused = (weight_s1 * s1_features) + (weight_s2 * s2_features)
        return fused


# --- Gating Encoder Model ---
class GatingUNet(nn.Module):
    """
    Dual-encoder U-Net with bottleneck gating.
    See Subsections 3.7.1 and 3.7.3.
    """
    def __init__(self, config):
        super(GatingUNet, self).__init__()
        
        self.num_classes = config['num_classes']
        self.encoder_name = config['encoder_name']
        self.encoder_weights = config['encoder_weights']
        
        # Initialize independent encoders for each modality 
        # See Subsection 3.7.1 (Dual Encoder Architecture)
        self.encoder_s1 = smp.encoders.get_encoder(
            name=self.encoder_name,
            in_channels=config['s1_channels'],
            weights=self.encoder_weights
        )
        
        self.encoder_s2 = smp.encoders.get_encoder(
            name=self.encoder_name,
            in_channels=config['s2_channels'],
            weights=self.encoder_weights
        )
        
        encoder_channels = self.encoder_s1.out_channels
        
        # Apply gating at the bottleneck
        # See Subsection 3.7.2 (Gating Mechanism)
        self.gating_module = GatingModule(
            feature_dim=encoder_channels[-1]
        )
        
        # Configure decoder channel sizes (doubling encoder output for skip connections)
        # See Equation 3.9
        combined_channels = [ch * 2 for ch in encoder_channels[:-1]] + [encoder_channels[-1]]
        
        # Decoder architecture
        # See Subsection 3.7.3 (Decoder, Skip Connection, and Segmentation Head)
        decoder_channels = [256, 128, 64, 32, 16]
        n_blocks = len(decoder_channels)
        
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=combined_channels,
            decoder_channels=decoder_channels,
            n_blocks=n_blocks,
        )
        
        # Spatial dropout to mitigate overfitting
        # See Subsection 3.7.3
        self.dropout = nn.Dropout2d(p=0.2)
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=self.num_classes,
            activation=None,
            kernel_size=3
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """
        Initializes weights for the decoder and segmentation head.
        """
        for name, m in self.named_modules():
            # Skip encoders to preserve designated initial states
            if "encoder" in name:
                continue
                
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Split input into S1 and S2 channels
        # See Equation 3.6
        s1_input = x[:, :4, :, :]
        s2_input = x[:, 4:, :, :]
        
        # Extract hierarchical features independently
        s1_features = self.encoder_s1(s1_input)
        s2_features = self.encoder_s2(s2_input)
        
        # Fuse bottleneck features via gating module
        s1_bottleneck = s1_features[-1]
        s2_bottleneck = s2_features[-1]
        fused_bottleneck = self.gating_module(s1_bottleneck, s2_bottleneck)

        # Concatenate features for skip connections
        combined_features = []
        for s1_feat, s2_feat in zip(s1_features, s2_features):
            combined = torch.cat([s1_feat, s2_feat], dim=1)
            combined_features.append(combined)

        # Replace the original bottleneck with the gated fusion output
        combined_features[-1] = fused_bottleneck

        decoder_output = self.decoder(combined_features)
        decoder_output = self.dropout(decoder_output)
        
        logits = self.segmentation_head(decoder_output)
        
        # Align output dimensions with input dimensions via bilinear interpolation
        # See Subsection 3.7.3
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, 
                size=x.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        return logits


# --- Factory Functions ---
def get_model(config):
    """
    Factory function to instantiate the GatingUNet and move it to the target device.
    """
    model = GatingUNet(config)
    device = config["device"]
    model = model.to(device)
    return model


def get_model_summary(model, input_shape):
    """
    Prints total and trainable parameter counts for the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input shape: {input_shape}")
    
    return total_params, trainable_params