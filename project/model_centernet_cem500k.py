"""CenterNet architecture with CEM500K ResNet50 encoder."""

import torch
import torch.nn as nn
import timm


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, features):
        """
        Args:
            features: List of feature maps from encoder (coarse to fine)

        Returns:
            List of FPN features (same scale, fine resolution)
        """
        # Build laterals
        laterals = [
            lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )

        # Apply FPN convs
        fpn_features = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]

        return fpn_features


class ConvHead(nn.Module):
    """Prediction head (conv layer)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CenterNetCEM500K(nn.Module):
    """CenterNet with CEM500K ResNet50 encoder."""

    def __init__(self, pretrained=True, freeze_encoder=True):
        super().__init__()

        # Encoder: CEM500K ResNet50 (pre-trained on 500K EM images)
        self.encoder = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get output channel sizes from encoder
        encoder_channels = [256, 512, 1024, 2048]
        fpn_channels = 256

        # FPN: Fuse multi-scale features
        self.fpn = FPN(encoder_channels, fpn_channels)

        # Decoder at 1/4 input stride (matches heatmap targets in dataset_centernet).
        # FPN finest map is ~H/4×W/4; do not upsample to full res (that broke loss vs 64×64 targets).
        self.decoder = nn.Sequential(
            DoubleConv(fpn_channels, 128),
            DoubleConv(128, 64),
            DoubleConv(64, 64),
        )

        # Prediction heads
        self.center_head = ConvHead(64, 1)
        self.class_head = ConvHead(64, 2)
        self.size_head = ConvHead(64, 2)
        self.offset_head = ConvHead(64, 2)
        self.confidence_head = ConvHead(64, 1)

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Dictionary with predictions at 1/4 stride
        """
        # Encoder
        features = self.encoder(x)  # List of 4 feature maps

        # FPN
        fpn_features = self.fpn(features)  # List of 4 FPN features

        # Use finest FPN feature (highest resolution)
        fused = fpn_features[0]

        # Decoder
        decoded = self.decoder(fused)

        # Prediction heads
        return {
            "centers": self.center_head(decoded),
            "classes": self.class_head(decoded),
            "sizes": self.size_head(decoded),
            "offsets": self.offset_head(decoded),
            "confidence": self.confidence_head(decoded),
        }

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self, num_blocks=2):
        """Unfreeze last num_blocks of encoder."""
        # ResNet50 structure: [layer1, layer2, layer3, layer4]
        blocks = [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]

        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def get_parameter_groups(self, lr_decoder=1e-4, lr_encoder=1e-5):
        """Get parameter groups for differential learning rates."""
        decoder_params = []
        encoder_params = []

        # Collect decoder and head parameters
        for name, param in self.named_parameters():
            if "encoder" not in name:
                decoder_params.append(param)
            else:
                encoder_params.append(param)

        return [
            {"params": decoder_params, "lr": lr_decoder},
            {"params": encoder_params, "lr": lr_encoder},
        ]


if __name__ == "__main__":
    model = CenterNetCEM500K(freeze_encoder=True)
    x = torch.randn(2, 3, 256, 256)
    output = model(x)

    print("Model outputs:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
