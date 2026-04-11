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
        try:
            self.backbone = AutoModel.from_pretrained(self.model_name)
        except Exception as e:
            from transformers import Dinov2Config
            # To fix the error on the submission server in which I could not load in the HF version
            # Found config on: https://huggingface.co/facebook/dinov2-large/blob/main/config.json 
            # #NOTE: Not included a seperate config.json file as I did not know if that would work on the submission server, so just hardcoded the config values here as a fallback
            config = Dinov2Config(hidden_size=1024,
                                  num_hidden_layers=24,
                                  num_attention_heads=16,
                                  patch_size=14,
                                  num_channels=3,
                                  attention_probs_dropout_prob=0.0,
                                  drop_path_rate=0.0,
                                  hidden_act="gelu",
                                  hidden_dropout_prob=0.0,
                                  layer_norm_eps=1e-06,
                                  layerscale_value=1.0,
                                  mlp_ratio=4,
                                  qkv_bias=True,
                                  use_swiglu_ffn=False,
                                  image_size=518) #NOTE: to fix positional embedding error
            self.backbone = AutoModel.from_config(config)

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

        # Fallback for other model sizes: evenly spaced quarters (as used in the DINOv3 and DINOv2 paper)
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
    NOTE: Uses a lower per-gpu bs (4 instead of 8) and a higher gradient accummulation (2 instead of 1). This makes the effective bs still 32 (similar to the other experiments). Done to prevent errors
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

        # Resize the final logits to exactly match the input image size, as the repeated upsampling not perfectly recovers the original resolution depending on the patch size and image size
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)


#Experiment 3: MLP decoder
class MLPHead(nn.Module):
    """
    True per-token MLP decoder head applied on the final DINOv2 patch feature map.

    Treats each patch token as an independent feature vector and applies a shared two-layer MLP across all tokens simultaneously.
    The spatial structure is temporarily collapsed for the linear operations and restored afterwards.
    This puts the parameter count between the linear head and the full upsampling decoder.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 n_output_channels: int):
        super().__init__()

        # Shared MLP applied independently to each patch token: in -> hidden -> hidden//2 -> out
        # LayerNorm is used instead of BatchNorm because we operate on the channel (feature) dimension per token, not on the spatial batch dimension 
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                 nn.LayerNorm(hidden_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels // 2),
                                 nn.LayerNorm(hidden_channels // 2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels // 2, n_output_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process each token independently with the shared MLP. The input is in (B, C, H, W) format, so reshape it to (B*H*W, C) to apply the MLP across all tokens at once.
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Apply the shared MLP to all tokens at once, then restore the spatial grid
        out = self.mlp(tokens)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class DinoMLPDecoder(DinoSegBase):
    """
    Lightweight pointwise MLP decoder on top of the fixed DINOv2 backbone.

    Operates on the final patch feature map only (no multi-level fusion, no spatial convolutions). 
    Sits between the linear head  and the upsampling decoder in terms of model size and capacity.
    """

    def _build_decoder(self) -> nn.Module:
        # hidden_channels=512 gives a moderate bottleneck for DINOv2-L (hidden_size=1024)
        return MLPHead(in_channels=self.hidden_size,
                       hidden_channels=512,
                       n_output_channels=self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # Use only the final patch feature map; same as the upsampling decoder
        feats = self.extract_patch_features(x)
        logits = self.decoder(feats)

        # Bilinear resize from patch resolution back to the full image resolution
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)



#NOTE: This is not used for the assignment anymore, as probably has data leakage. Training it from scratch was too difficult, so swtiched to something else.
#NOTE: Still submitted on the submission server, but not included in the report.
# Experiment 4: EoMT-DINOv2 (encoder-only)
class DinoEoMT(nn.Module):
    """
    Wrapper around the Hugging Face EoMT semantic segmentation model built on a DINOv2-L backbone.
    
    EoMT stems from: "Your ViT is Secretly an Image Segmentation Model" (Kerssies et al., CVPR 2025) arXiv:2503.19108. 
    
    The used checkpoint (tue-mps/cityscapes_semantic_eomt_large_1024) was trained at 1024x1024, while Cityscapes images are 1024x2048. So use sliding window in predicting
    """
    def __init__(self, model_name: str | None = None, n_classes: int = 19):
        super().__init__()
        
        # Load the pretrained EoMT segmentation model from Hugging Face
        self.n_classes = n_classes
        self.model_name = eomt_backbone_name
        self.model = AutoModelForUniversalSegmentation.from_pretrained(self.model_name)
        self.window_size = 1024
        self.window_stride = 768
        
        # Keep a reference to the underlying backbone so the training script can freeze/unfreeze it if needed
        self.backbone = getattr(self.model, "model", self.model)
        
        # If the pretrained segmentation head predicts a different number of classes than Cityscapes, adapt it with a 1x1 projection
        if self.model.config.num_labels != self.n_classes:
            self.class_adapter = nn.Conv2d(self.model.config.num_labels, self.n_classes, kernel_size=1)
        else:
            self.class_adapter = nn.Identity()

    def _forward_window(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the pretrained EoMT model on a single 1024x1024 crop or window."""
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

    def _get_window_starts(self, image_size: int) -> list[int]:
        """Builds overlapping window start indices and ensures the final window reaches the image border."""
        if image_size <= self.window_size:
            return [0]

        starts = list(range(0, image_size - self.window_size + 1, self.window_stride))
        last_start = image_size - self.window_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _forward_sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        """Runs overlapping sliding-window inference and averages logits in overlap regions."""
        batch_size, _, height, width = x.shape
        output_sum = torch.zeros((batch_size, self.n_classes, height, width), dtype=torch.float32, device=x.device)
        output_count = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=x.device)

        y_starts = self._get_window_starts(height)
        x_starts = self._get_window_starts(width)

        # Process each image independently so the shared training loop can still pass a full batch into model(images).
        for batch_index in range(batch_size):
            image = x[batch_index:batch_index + 1]

            for y0 in y_starts:
                y1 = min(y0 + self.window_size, height)
                for x0 in x_starts:
                    x1 = min(x0 + self.window_size, width)

                    window = image[:, :, y0:y1, x0:x1]
                    window_logits = self._forward_window(window).float()

                    output_sum[batch_index:batch_index + 1, :, y0:y1, x0:x1] += window_logits
                    output_count[batch_index:batch_index + 1, :, y0:y1, x0:x1] += 1.0

        return output_sum / output_count.clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The pretrained Cityscapes EoMT checkpoint expects 1024x1024 views.
        # For full-resolution Cityscapes inference, run overlapping sliding-window inference and average the logits in overlap regions.
        height, width = x.shape[-2:]
        if height > self.window_size or width > self.window_size:
            return self._forward_sliding_window(x)

        return self._forward_window(x)


# Submission entry point; predict.py imports and instantiates this class by name
class Model(DinoSegBase):
    """
    Concrete backbone + decoder model used by predict.py for submission.

    The decoder argument must match the one used during training so that
    load_state_dict(strict=True) restores the weights without any key mismatches.

    Parameters
        decoder : str
            Which decoder head to attach. One of 'linear', 'upsample', or 'mlp'.
        n_classes : int
            Number of output segmentation classes (19 for Cityscapes).
    """

    valid_decoders = ("linear", "upsample", "mlp")

    def __init__(self,
                 decoder = "upsample",
                 n_classes = 19):
        
        # Safeguard so it does not accidentally loads EoMT
        if decoder not in self.valid_decoders:
            raise ValueError(f"Unknown decoder '{decoder}'. Choose from: {self.valid_decoders}")

        self._decoder_type = decoder
        super().__init__(n_classes=n_classes)

    def _build_decoder(self) -> nn.Module:
        # Mirror the exact architecture from the corresponding training subclass; self.hidden_size and self.n_classes are already set by DinoSegBase.__init__ at this point.
        #NOTE: We do not use the already defined training subclasses to prevent loading the backbone twice. The Model class is solely for doing inference on the test set (submission server)
        if self._decoder_type == "linear":
            n_feature_maps = len(self.get_linear_feature_layer_indices())
            return LinearHead(in_channels=[self.hidden_size] * n_feature_maps,
                              n_output_channels=self.n_classes,
                              resize_factors=None,
                              use_batchnorm=True)

        elif self._decoder_type == "upsample":
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(self.hidden_size, 512),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(512, 256),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(256, 128),
                                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 ConvBlock(128, 64),
                                 nn.Conv2d(64, self.n_classes, kernel_size=1))

        elif self._decoder_type == "mlp":
            return MLPHead(in_channels=self.hidden_size,
                           hidden_channels=512,
                           n_output_channels=self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # The linear decoder fuses four intermediate feature maps from different backbone depths.
        # The upsample and mlp decoders both work on only the final patch feature map.
        if self._decoder_type == "linear":
            feats = self.extract_intermediate_features(x, layer_indices=self.get_linear_feature_layer_indices())
        else:
            feats = self.extract_patch_features(x)

        logits = self.decoder(feats)

        # Resize from patch-grid resolution back to the original image size
        return F.interpolate(logits,
                             size=(H, W),
                             mode="bilinear",
                             align_corners=False)
