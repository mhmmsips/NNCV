import torch
import torch.nn as nn
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoSegBase(nn.Module):
    """
    Base class for all the used DINO-based segmentation models. Handles backbone loading and patch token extraction.
    Subclasses only need to implement _build_decoder() and forward().
    """
    def __init__(self, model_name: str, n_classes: int = 19):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        self.decoder = self._build_decoder()
        
    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 2D spatial feature map from backbone patch tokens."""
        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state # (B, 1+N, C)
        
        n_registers = getattr(self.backbone.config, 'num_register_tokens', 0)
        patch_tokens = tokens[:, 1 + n_registers:, :]  # drop CLS + register tokens #NOTE: This is needed because in ViT the output tokens contain one extra CLS token
        
        B, N, C = patch_tokens.shape # (B, N, C)
        grid_size = int(N ** 0.5) # Assume square grid of patches
        
        # Sanity check: N should be a perfect square for reshaping into (h, w)
        assert grid_size ** 2 == N, f"Patch count {N} is not a perfect square"

        # (B, N, C) -> (B, C, grid, grid)
        return patch_tokens.permute(0, 2, 1).contiguous().view(B, C, grid_size, grid_size)


    def _build_decoder(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_decoder()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")


# Decoder 1: DINOv2 styled linear head
class DinoLinearDecoder(DinoSegBase):
    """
    DINO-style linear segmentation head.
    Equivalent to the DINOv2 BNHead: BatchNorm -> 1x1 Conv.
    """

    def _build_decoder(self) -> nn.Module:
        return nn.Sequential(nn.BatchNorm2d(self.hidden_size),
                             nn.Conv2d(self.hidden_size, self.n_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        feats = self.extract_patch_features(x) # (B, C, h, w)

        logits = self.decoder(feats) # (B, n_classes, h, w)

        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)

# Decoder 2: MLP
class DinoMLPDecoder(DinoSegBase):
    """
    3-layer MLP segmentation head applied per patch token.
    """

    def _build_decoder(self) -> nn.Module:
        hidden_dim = 2 * self.hidden_size

        return nn.Sequential(nn.Linear(self.hidden_size, hidden_dim),
                             nn.GELU(), #NOTE: Transformers (BERT, ViT, DINO, CLIP, etc.) almost always use GELU inside the MLP blocks. DINO backbone was also trained with GELU's everywhere.
                             nn.Linear(hidden_dim, hidden_dim),
                             nn.GELU(), #NOTE: Transformers (BERT, ViT, DINO, CLIP, etc.) almost always use GELU inside the MLP blocks. DINO backbone was also trained with GELU's everywhere.
                             nn.Linear(hidden_dim, self.n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = self.extract_patch_features(x) # (B, C, h, w)

        B, C, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, h * w, C)  # (B, N, C)

        logits = self.decoder(feats) # (B, N, n_classes)

        logits = logits.reshape(B, h, w, self.n_classes).permute(0, 3, 1, 2) # (B, n_classes, h, w)

        return F.interpolate(logits, 
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)



# Decoder 3: Progressive upsampling conv decoder
class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), #
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DinoUpsamplingDecoder(DinoSegBase):
    """
    Progressive upsampling convolutional decoder on top of DINO patch features.
    Not a true U-Net, because there are no encoder-decoder skip connections; but inspired by the U-Net decoder design with repeated upsampling and conv blocks.
    """

    def _build_decoder(self) -> nn.Module:
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(self.hidden_size, 1024),
                             
                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(1024, 512),
                             
                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(512, 256),
                             
                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(256, 128),
                             
                             nn.Conv2d(128, self.n_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = self.extract_patch_features(x) # (B, C, h, w)
        logits = self.decoder(feats)

        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)