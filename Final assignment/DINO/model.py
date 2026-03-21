import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForUniversalSegmentation


# Define the fixed DINOv2 backbone and the EoMT checkpoint to use
#NOTE: The main backbone experiments now use plain DINOv2-L/14, while EoMT is treated as its own pretrained segmentation model
#NOTE: DINOv3 version of EoMT is available but does not have a checkpoint for Cityscapes. Also training that might be out of the scope for the course.
dino_v2_backbone_name = "facebook/dinov2-large"
eomt_backbone_name = "tue-mps/cityscapes_semantic_eomt_large_1024"


class DinoSegBase(nn.Module):
    """
    Base class for the used DINOv2-based segmentation models.
    
    Subclasses only need to implement _build_decoder() and forward().
    """
    def __init__(self, model_name: str | None = None, n_classes: int = 19):
        super().__init__()
        
        # Store the number of output classes and always load the fixed DINOv2 backbone
        self.n_classes = n_classes
        self.model_name = dino_v2_backbone_name
        self.backbone = AutoModel.from_pretrained(self.model_name)
        
        # Extract the backbone metadata that is needed by the decoders
        self.hidden_size = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        
        # Let the subclass create the task-specific decoder on top of the backbone
        self.decoder = self._build_decoder()

    def _tokens_to_feature_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Converts patch tokens to a 2D feature map."""
        # DINOv2 outputs one CLS token. Register-token variants insert those tokens before the patch tokens, so skip them as well when reshaping the dense patch descriptors back to a 2D feature map
        # For dense prediction only keep the patch tokens and reshape them back to a 2D grid 
        n_registers = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_tokens = tokens[:, 1 + n_registers:, :]
        
        B, N, C = patch_tokens.shape
        grid_height = image_size[0] // self.patch_size
        grid_width = image_size[1] // self.patch_size
        
        # Cityscapes images are rectangular, so the patch grid must be rectangular as well
        # Recover the spatial grid directly from the input size and the patch size instead of assuming a square grid
        assert grid_height * grid_width == N, f"Patch grid {grid_height}x{grid_width} does not match token count {N}"

        # Convert from (B, N, C) to (B, C, H, W), which is the format expected by convolutional decoder heads
        return patch_tokens.permute(0, 2, 1).contiguous().view(B, C, grid_height, grid_width)

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the final 2D patch feature map from the fixed DINOv2 backbone."""
        # Run the backbone once and use only the final hidden state for decoder heads that expect a single feature map
        outputs = self.backbone(pixel_values=x)
        return self._tokens_to_feature_map(outputs.last_hidden_state, x.shape[-2:])

    def get_linear_feature_layer_indices(self, n_feature_maps: int = 4) -> list[int]:
        """Selects backbone layer indices for the linear decoder head, following the DINOv2 paper for known model sizes."""
        total_layers = self.backbone.config.num_hidden_layers
        n_feature_maps = min(n_feature_maps, total_layers)

        # Exact indices from the DINOv2 paper ("DINOv2: Learning Robust Visual Features without Supervision" arXiv:2304.07193) for the four standard model sizes
        paper_indices = {12: [3, 6, 9, 12], # ViT-S/14 and ViT-B/14
                         24: [5, 12, 18, 24], # ViT-L/14
                         40: [10, 20, 30, 40] # ViT-g/14
                         }
        
        if total_layers in paper_indices and n_feature_maps == 4:
            return paper_indices[total_layers]

        # Fallback for other model sizes: evenly spaced quarters (as used in the DINOv3 paper)
        step = total_layers / n_feature_maps
        return [round(i * step) for i in range(1, n_feature_maps + 1)]

    def extract_intermediate_features(self, x: torch.Tensor, layer_indices: list[int] | None = None) -> list[torch.Tensor]:
        """Extracts the selected intermediate patch feature maps from the fixed DINOv2 backbone."""
        # Request all hidden states so the linear decoder can fuse feature maps from multiple depths of the transformer
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        #NOTE: hidden_states[0] is the embedding output, so the actual transformer blocks start at index 1
        if layer_indices is None:
            layer_indices = self.get_linear_feature_layer_indices()

        return [self._tokens_to_feature_map(hidden_states[layer_index], x.shape[-2:]) for layer_index in layer_indices]

    
    def _build_decoder(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _build_decoder()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")


# Experiment 1: Linear decoder
class LinearHead(nn.Module):
    """
    BatchNorm-based linear decoder head on top of multi-level DINOv2 features.
    
    The implementation is functionally inspired by Meta's BNHead for DINOv2 semantic segmentation.
    
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/models/decode_heads/linear_head.py
    """

    def __init__(self,
                 in_channels,
                 n_output_channels,
                 resize_factors=None,
                 use_batchnorm=True):
        
        super().__init__()
        
        # Store the per-level input channel sizes and compute the fused channel count after concatenation
        self.in_channels = list(in_channels)
        self.channels = sum(in_channels)
        self.n_output_channels = n_output_channels
        self.resize_factors = resize_factors
        
        # Apply normalization and a 1x1 classifier after the multi-level feature maps have been fused
        self.batchnorm_layer = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity()
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        
        # Initialize the segmentation classifier in the same way as the official Meta implementation
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _resize_feature_map(self, x: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Applies the optional pre-concatenation resizing used by the DINOv2 BN head."""
        if scale_factor == 1:
            return x

        if scale_factor >= 1:
            return F.interpolate(x,
                                 scale_factor=scale_factor,
                                 mode="bilinear",
                                 align_corners=False)

        return F.interpolate(x,
                             scale_factor=scale_factor,
                             mode="area")

    def _transform_inputs(self, inputs):
        """Transforms and fuses multi-level decoder inputs for the decoder."""
        # Initialize list of input feature maps
        inputs = list(inputs)

        # Ensure every selected feature map is in 4D (B, C, H, W) format
        for i, x in enumerate(inputs):
            if len(x.shape) == 2:
                x = x[:, :, None, None]
            inputs[i] = x

        # The official DINOv2 BN head optionally rescales the selected feature maps before concatenation.
        # That is kept as functionality here, even though the plain backbone already returns all selected layers on the same patch grid
        if self.resize_factors is not None:
            assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
            inputs = [self._resize_feature_map(x, scale_factor) for x, scale_factor in zip(inputs, self.resize_factors)]

        # Resize all feature maps to the spatial resolution of the first one, then concatenate them along the channel dimension.
        inputs = [F.interpolate(input=x,
                                size=inputs[0].shape[2:],
                                mode="bilinear",
                                align_corners=False) for x in inputs]
        
        return torch.cat(inputs, dim=1)

    def _forward_feature(self, inputs):
        """Applies the feature preprocessing before the final classifier."""
        # Fuse the selected transformer feature maps, then normalize them before the pixel classifier
        x = self._transform_inputs(inputs)
        return self.batchnorm_layer(x)

    def forward(self, inputs):
        """Forward pass through the linear decoder head."""
        # Fuse the multi-level features and predict per-pixel logits
        output = self._forward_feature(inputs)
        output = self.conv(output)
        return output


class DinoLinearDecoder(DinoSegBase):
    """
    Linear segmentation head on top of the fixed DINOv2 backbone.
    """

    def _build_decoder(self) -> nn.Module:
        # Use the selected intermediate feature maps from the backbone and fuse them with the linear head implementation inspired by the DINOv2 BN head design
        return LinearHead(in_channels=[self.hidden_size] * len(self.get_linear_feature_layer_indices()),
                          n_output_channels=self.n_classes,
                          resize_factors=None,
                          use_batchnorm=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the original input resolution so the final logits can be resized back to the image size
        H, W = x.shape[-2:]
        
        # Extract the evenly spread hidden states from the backbone and pass them to the linear decoder
        feats = self.extract_intermediate_features(x, layer_indices=self.get_linear_feature_layer_indices())
        logits = self.decoder(feats)
        
        # Resize the logits from patch resolution back to the full image resolution
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)



# Experiment 2: Progressive upsampling decoder
class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Use a standard double-convolution block for each upsampling stage.
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DinoUpsamplingDecoder(DinoSegBase):
    """
    Progressive upsampling convolutional decoder on top of DINOv2 patch features.
    Not a true U-Net, because there are no encoder-decoder skip connections; but inspired by the U-Net decoder design with repeated upsampling and conv blocks.
    
    NOTE: Aimed at matching the number of parameters from the U-net decoder (7.9M), where this one has 9.4M for DinoV2 (so this is slightly bigger)
    """

    def _build_decoder(self) -> nn.Module:
        # Start from the low-resolution DINO patch feature map and progressively upsample it with convolution blocks
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(self.hidden_size, 512),

                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(512, 256),

                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(256, 128),

                             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             ConvBlock(128, 64),

                             nn.Conv2d(64, self.n_classes, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        
        # Use only the final patch feature map for the upsampling decoder
        feats = self.extract_patch_features(x)
        logits = self.decoder(feats)

        # Resize the final logits to exactly match the input image size
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)


# Experiment 3: EoMT-DINOv2 (encoder-only)
class DinoEoMT(nn.Module):
    """
    Wrapper around the Hugging Face EoMT semantic segmentation model built on a DINOv2-L backbone.
    
    EoMT stems from: "Your ViT is Secretly an Image Segmentation Model" (Kerssies et al., CVPR 2025) arXiv:2503.19108. 
    
    The used checkpoint (tue-mps/cityscapes_semantic_eomt_large_1024) was trained at 1024x1024, while Cityscapes images are 1024x2048.
    That sounds problematic at first glance, but the published 84.2 mIoU (See: https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov2.md) number was measured on the actual Cityscapes validation set. 
    In other words, whatever padding, cropping, or resizing the official EoMT evaluation pipeline applies to handle the aspect-ratio mismatch is already reflected in that benchmark number)
    """
    def __init__(self, model_name: str | None = None, n_classes: int = 19):
        super().__init__()
        

        # Load the pretrained EoMT segmentation model from Hugging Face
        self.n_classes = n_classes
        self.model_name = eomt_backbone_name
        self.model = AutoModelForUniversalSegmentation.from_pretrained(self.model_name)
        
        # Keep a reference to the underlying backbone so the training script can freeze/unfreeze it if needed
        self.backbone = getattr(self.model, "model", self.model)
        
        # If the pretrained segmentation head predicts a different number of classes than Cityscapes, adapt it with a 1x1 projection
        if self.model.config.num_labels != self.n_classes:
            self.class_adapter = nn.Conv2d(self.model.config.num_labels, self.n_classes, kernel_size=1)
        else:
            self.class_adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        
        # Run the EoMT model to obtain query classification scores and mask logits
        outputs = self.model(pixel_values=x)

        # Convert the query outputs into dense per-class segmentation logits by weighting the masks with the class probabilities
        class_scores = outputs.class_queries_logits[..., :-1].softmax(dim=-1)
        mask_scores = outputs.masks_queries_logits.sigmoid()
        logits = torch.einsum("bqc,bqhw->bchw", class_scores, mask_scores)
        logits = self.class_adapter(logits)

        # Resize the final dense logits to the original image resolution
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)
