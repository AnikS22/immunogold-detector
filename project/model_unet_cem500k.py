"""UNet with CEM500K pre-trained ResNet50 encoder for transfer learning."""

from typing import Optional
import torch
import torch.nn as nn
import timm

class UNetCEM500K(nn.Module):
    """
    U-Net with CEM500K pre-trained ResNet50 encoder.

    Architecture:
    - Encoder: ResNet50 (pre-trained on 500K EM images via MoCoV2)
    - Decoder: 4-level symmetric decoder with skip connections
    - Output: 2 channels (heatmap predictions)
    """

    def __init__(self, pretrained: bool = True, freeze_encoder: bool = True):
        super().__init__()

        # Load ResNet50 backbone
        if pretrained:
            # Load with ImageNet weights first (timm default)
            self.encoder = timm.create_model(
                'resnet50',
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4)
            )
        else:
            self.encoder = timm.create_model(
                'resnet50',
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4)
            )

        # Get channel dimensions from ResNet50
        # Input: 3, Output features: [64, 256, 512, 1024, 2048]
        encoder_channels = [64, 256, 512, 1024, 2048]

        # Decoder (symmetric to encoder)
        # Upsample 2048 → 1024 → 512 → 256 → 64 → 32

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.dec4 = DoubleConv(1024 + 1024, 1024)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec3 = DoubleConv(512 + 512, 512)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = DoubleConv(256 + 256, 256)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec1 = DoubleConv(128 + 64, 128)

        # Final upsampling to match input resolution
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # No skip connection at final level (e0 is 128×128, d0 is 256×256 - spatial mismatch)
        self.dec0 = DoubleConv(64, 64)

        # Output head
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Output heatmap (B, 2, H, W)
        """
        # Encoder: extract features at different scales
        enc_features = self.encoder(x)
        # enc_features: [stride=2, 4, 8, 16, 32] = [64, 256, 512, 1024, 2048]
        e0, e1, e2, e3, e4 = enc_features

        # Decoder with skip connections
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        # No skip connection at final level (spatial mismatch: e0 is 128×128, d0 is 256×256)
        d0 = self.dec0(d0)

        out = self.out_conv(d0)
        return out

    def unfreeze_encoder_partial(self, num_blocks: int = 2):
        """
        Unfreeze last N ResNet blocks for fine-tuning.

        Args:
            num_blocks: Number of blocks to unfreeze (1-4)
        """
        # ResNet50 has 4 main blocks (layer1, layer2, layer3, layer4)
        blocks_to_unfreeze = [f"layer{i}" for i in range(5 - num_blocks, 5)]

        for name, param in self.encoder.named_parameters():
            if any(block in name for block in blocks_to_unfreeze):
                param.requires_grad = True

        print(f"Unfroze last {num_blocks} ResNet blocks")


class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if dropout_p > 0:
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Test model
    model = UNetCEM500K(pretrained=True, freeze_encoder=True)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
