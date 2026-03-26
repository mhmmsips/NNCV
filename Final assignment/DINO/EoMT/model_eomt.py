import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, EomtConfig, EomtForUniversalSegmentation


# Keep the backbone fixed to the same plain self-supervised DINOv2-L/14 model that is used in the rest of the comparison.
dino_v2_backbone_name = "facebook/dinov2-large"


class Model(nn.Module):
    """
    EoMT-L semantic segmentation model built from config and initialized with plain DINOv2-L/14 encoder weights.

    NOTE: The segmentation-specific parts of EoMT stay randomly initialized on purpose. Only the ViT encoder weights are transferred from DINOv2-L.
    """
    def __init__(self,
                 in_channels=3,
                 n_classes=19):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.model_name = dino_v2_backbone_name

        # Load the fixed DINOv2 backbone first, because EoMT should only reuse these self-supervised encoder weights and nothing segmentation-specific.
        dino_backbone = AutoModel.from_pretrained(self.model_name,
                                                  attn_implementation="sdpa")
        
        backbone_config = dino_backbone.config

        self.patch_size = backbone_config.patch_size
        self.hidden_size = backbone_config.hidden_size
        self.num_queries = 100

        # The official EoMT repo keeps the last 4 ViT blocks active for mask prediction by default. 
        # The Cityscapes semantic config overrides `num_q: 100`, but does not override the number of active blocks, so still uses 4 here.
        self.num_blocks = 4

        # In the official repo the number of upscale blocks is derived from the patch size instead of being written out manually. 
        # For a 14x14 DINOv2 backbone this gives 1 upscale block: max(1, int(log2(14)) - 2) = 1
        self.num_upscale_blocks = max(1, int(math.log2(self.patch_size)) - 2)

        # Ensure that we do not get errors during training because of a mismatched patch size
        if self.patch_size != 14:
            raise ValueError(f"Expected a 14x14 DINOv2 patch size, but got {self.patch_size}")

        # EoMT in transformers (hf) is documented for square image sizes. 
        #NOTE: Cityscapes is rectangular, so build the model once with a harmless square size and then replace the patch-grid metadata and positional embeddings with the correct rectangular Cityscapes version.
        cityscapes_padded_height = self._get_padded_size(1024)
        cityscapes_padded_width = self._get_padded_size(2048)

        # Build the Hugging Face EoMT config in two steps:
        # First: Copy the plain ViT-L/14 backbone shape from `facebook/dinov2-large`.
        # Second: Fill in the EoMT-specific segmentation settings so the architecture matches the official repo as closely as possible, without loading a retrained segmentation checkpoint.
        
        eomt_config = EomtConfig(
            # hidden_size / layer count / heads / patch_size / MLP ratio / etc. come straight from the DINOv2-L backbone config.
            hidden_size=backbone_config.hidden_size,
            num_hidden_layers=backbone_config.num_hidden_layers,
            num_attention_heads=backbone_config.num_attention_heads,
            mlp_ratio=backbone_config.mlp_ratio,
            hidden_act=backbone_config.hidden_act,
            hidden_dropout_prob=backbone_config.hidden_dropout_prob,
            initializer_range=backbone_config.initializer_range,
            layer_norm_eps=backbone_config.layer_norm_eps,
            patch_size=backbone_config.patch_size,
            layerscale_value=backbone_config.layerscale_value,
            drop_path_rate=backbone_config.drop_path_rate,
            num_channels=in_channels,
            
            # image_size is a temporary Hugging Face config value. EoMT in the repo handles rectangular image sizes directly, but the Hugging Face config wants an image size up front, so we patch the real rectangular grid a few lines later.
            image_size=cityscapes_padded_height,
            
            # num_upscale_blocks is derived from patch size in the same way as the official repo, which gives 1 for patch size 14.
            num_upscale_blocks=self.num_upscale_blocks,
            
            #num_blocks=4 follows the default in the official `models/eomt.py`.
            num_blocks=self.num_blocks,
            
            # num_queries=100 comes from the official Cityscapes config.
            num_queries=self.num_queries,
            
            # num_register_tokens comes from the chosen backbone (0 for `facebook/dinov2-large`)
            num_register_tokens=getattr(backbone_config, "num_register_tokens", 0),
            
            # number of classes of Cityscapes to predict
            num_labels=n_classes)

        # Build the full EoMT architecture from config. This keeps the query embeddings, mask head and classification head random, which is what we want for a fair comparison against Linear and Upsample.
        self.model = EomtForUniversalSegmentation(eomt_config)

        #NOTE: Look into FlashAttention when time allows it
        self.model.set_attn_implementation("sdpa")
        
        self.backbone = self.model

        # Set the backbone state dict from DINOv2 and extract the patch positional embeddings to use as a reference for resizing to the Cityscapes grid. 
        dino_state_dict = dino_backbone.state_dict()
        dino_patch_position_embeddings, dino_position_grid = self._extract_patch_position_embeddings(dino_state_dict)

        # Set the correct rectangular grid and resized position embeddings for Cityscapes
        self._set_rectangular_grid(padded_height=cityscapes_padded_height,
                                   padded_width=cityscapes_padded_width,
                                   reference_position_embeddings=dino_patch_position_embeddings,
                                   reference_grid_size=dino_position_grid,)

        # Map the DINOv2 state dict to the corresponding EoMT parameters and load it with strict=False, then validate that all the expected backbone weights were loaded and that no unexpected non-backbone weights were loaded.
        mapped_dino_state_dict = self._map_dino_state_dict(dino_state_dict)
        load_result = self.model.load_state_dict(mapped_dino_state_dict, strict=False)
        self._validate_backbone_weight_loading(load_result, mapped_dino_state_dict)

        # Clean up, because the DINOv2 backbone is not needed anymore and is large in memory
        del dino_backbone


    def _get_padded_size(self, size):
        # Function to compute the padded image size that is divisible by the patch size, given an original image size
        return math.ceil(size / self.patch_size) * self.patch_size


    def _extract_patch_position_embeddings(self, dino_state_dict):
        # DINOv2 stores one positional embedding for the CLS token and one for each patch. 
        # EoMT keeps the CLS token separate, so only keep the patch part here.
        position_embeddings = None
        for key in ("embeddings.position_embeddings", "dinov2.embeddings.position_embeddings"):
            if key in dino_state_dict:
                position_embeddings = dino_state_dict[key]
                break
        
        # Some safeguards
        if position_embeddings is None:
            raise KeyError("Could not find DINOv2 positional embeddings in the backbone state dict")
        if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
            raise ValueError(f"Unexpected DINOv2 positional embedding shape: {tuple(position_embeddings.shape)}")

        # Remove the CLS token embedding and the leading batch dimension, leaving only the patch position embeddings in a (num_patches, hidden_size) tensor. 
        # Also infer the original DINOv2 patch grid size from the number of patch position embeddings.
        patch_position_embeddings = position_embeddings[:, 1:, :].squeeze(0)
        num_patch_positions = patch_position_embeddings.shape[0]
        source_grid_side = math.isqrt(num_patch_positions)

        # Safeguard against non-square grid
        if source_grid_side * source_grid_side != num_patch_positions:
            raise ValueError(f"Expected a square DINOv2 patch grid, but got {num_patch_positions} patch positions")

        return patch_position_embeddings, (source_grid_side, source_grid_side)


    def _map_dino_state_dict(self, dino_state_dict):
        
        # Initialize dict to hold mapped DINOv2 weights that correspond to EoMT parameters
        mapped_state_dict = {}

        attention_name_map = {"attention.attention.query.": "attention.q_proj.",
                              "attention.attention.key.": "attention.k_proj.",
                              "attention.attention.value.": "attention.v_proj.",
                              "attention.output.dense.": "attention.out_proj."}

        # Loop over the state_dict keys and map the DINOv2 ViT encoder weights to the corresponding EoMT parameters
        for original_key, value in dino_state_dict.items():
            key = original_key.removeprefix("dinov2.")

            # EoMT uses a different positional embedding layout, and this model does not use DINOv2's masked-image-modeling token.
            if key in {"embeddings.mask_token", "embeddings.position_embeddings"}:
                continue

            if key.startswith("encoder.layer."):
                key = "layers." + key[len("encoder.layer."):]

            for source_name, target_name in attention_name_map.items():
                if source_name in key:
                    key = key.replace(source_name, target_name)
                    break

            mapped_state_dict[key] = value

        return mapped_state_dict

    def _interpolate_patch_position_embeddings(self,
                                               position_embeddings: torch.Tensor,
                                               source_grid_size: tuple[int, int],
                                               target_grid_size: tuple[int, int]) -> torch.Tensor:
        if source_grid_size == target_grid_size:
            return position_embeddings

        source_height, source_width = source_grid_size
        target_height, target_width = target_grid_size
        hidden_size = position_embeddings.shape[-1]
        original_dtype = position_embeddings.dtype

        position_embeddings = position_embeddings.reshape(1, source_height, source_width, hidden_size).permute(0, 3, 1, 2)
        position_embeddings = position_embeddings.to(dtype=torch.float32)

        try:
            # NOTE: Positional embeddings are smooth learned features on a 2D grid, so bicubic interpolation is a sensible way to resize them to a new patch grid.
            position_embeddings = F.interpolate(position_embeddings,
                                                size=(target_height, target_width),
                                                mode="bicubic",
                                                align_corners=False,
                                                antialias=True)
        except TypeError:
            position_embeddings = F.interpolate(position_embeddings,
                                                size=(target_height, target_width),
                                                mode="bicubic",
                                                align_corners=False)

        position_embeddings = position_embeddings.permute(0, 2, 3, 1).reshape(target_height * target_width, hidden_size)
        return position_embeddings.to(dtype=original_dtype)

    def _set_rectangular_grid(self,
                              padded_height: int,
                              padded_width: int,
                              reference_position_embeddings: torch.Tensor | None = None,
                              reference_grid_size: tuple[int, int] | None = None) -> None:
        target_grid_size = (padded_height // self.patch_size, padded_width // self.patch_size)
        num_patch_positions = target_grid_size[0] * target_grid_size[1]

        # If a new image size is seen later, resize from the current learned
        # position table rather than from the original DINOv2 one.
        if reference_position_embeddings is None or reference_grid_size is None:
            reference_position_embeddings = self.model.embeddings.position_embeddings.weight.detach()
            reference_grid_size = tuple(self.model.grid_size)

        interpolated_positions = self._interpolate_patch_position_embeddings(
            position_embeddings=reference_position_embeddings,
            source_grid_size=reference_grid_size,
            target_grid_size=target_grid_size,
        )

        embedding_device = self.model.embeddings.patch_embeddings.projection.weight.device
        embedding_dtype = self.model.embeddings.patch_embeddings.projection.weight.dtype
        interpolated_positions = interpolated_positions.to(device=embedding_device,
                                                           dtype=embedding_dtype)

        new_position_embeddings = nn.Embedding(num_patch_positions,
                                               interpolated_positions.shape[-1]).to(device=embedding_device,
                                                                                    dtype=embedding_dtype)

        with torch.no_grad():
            new_position_embeddings.weight.copy_(interpolated_positions)

        self.model.embeddings.position_embeddings = new_position_embeddings
        self.model.embeddings.position_ids = torch.arange(num_patch_positions,
                                                          device=embedding_device).expand((1, -1))
        self.model.grid_size = target_grid_size
        self.model.embeddings.patch_embeddings.image_size = (padded_height, padded_width)
        self.model.embeddings.patch_embeddings.num_patches = num_patch_positions

    def _validate_backbone_weight_loading(self,
                                          load_result,
                                          mapped_dino_state_dict: dict[str, torch.Tensor]) -> None:
        encoder_prefixes = ("embeddings.", "layers.", "layernorm.")
        allowed_missing_prefixes = (
            "embeddings.position_embeddings",
            "embeddings.position_ids",
            "embeddings.register_tokens",
        )

        mapped_encoder_keys = sorted(
            key for key in mapped_dino_state_dict
            if key.startswith(encoder_prefixes)
        )
        if not mapped_encoder_keys:
            raise ValueError("No encoder-compatible DINOv2 weights were found for EoMT")

        unexpected_encoder_keys = sorted(
            key for key in load_result.unexpected_keys
            if key.startswith(encoder_prefixes)
        )
        if unexpected_encoder_keys:
            raise ValueError(
                "Some mapped DINOv2 encoder weights did not match EoMT encoder parameters: "
                f"{unexpected_encoder_keys[:20]}"
            )

        missing_encoder_keys = sorted(
            key for key in load_result.missing_keys
            if key.startswith(encoder_prefixes) and not key.startswith(allowed_missing_prefixes)
        )
        if missing_encoder_keys:
            raise ValueError(
                "Some EoMT encoder parameters were not initialized from DINOv2 as expected: "
                f"{missing_encoder_keys[:20]}"
            )

    def _extract_prediction_pairs(self, outputs) -> list[tuple[torch.Tensor, torch.Tensor]]:
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        if isinstance(class_queries_logits, (list, tuple)):
            if not isinstance(masks_queries_logits, (list, tuple)):
                raise TypeError(
                    "class_queries_logits is a list/tuple but masks_queries_logits is not"
                )

            if len(class_queries_logits) != len(masks_queries_logits):
                raise ValueError("class_queries_logits and masks_queries_logits do not have the same length")

            return list(zip(class_queries_logits, masks_queries_logits))

        return [(class_queries_logits, masks_queries_logits)]

    def _query_predictions_to_dense_logits(self,
                                           class_queries_logits: torch.Tensor,
                                           masks_queries_logits: torch.Tensor,
                                           padded_height: int,
                                           padded_width: int) -> torch.Tensor:
        # Convert the EoMT query predictions to dense semantic logits using the
        # MaskFormer-style class-probability × mask-probability combination.
        masks_full_resolution = F.interpolate(masks_queries_logits,
                                              size=(padded_height, padded_width),
                                              mode="bilinear",
                                              align_corners=False).sigmoid()

        class_probabilities = class_queries_logits[..., :-1].softmax(dim=-1)
        return torch.einsum("bqc,bqhw->bchw", class_probabilities, masks_full_resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        original_height, original_width = x.shape[-2:]
        padded_height = self._get_padded_size(original_height)
        padded_width = self._get_padded_size(original_width)
        pad_bottom = padded_height - original_height
        pad_right = padded_width - original_width

        # NOTE: When train_eomt.py is used, the transforms already pad the images to 1036x2058
        if pad_bottom > 0 or pad_right > 0:
            x = F.pad(x,
                      (0, pad_right, 0, pad_bottom),
                      mode="constant",
                      value=0.0)

        current_grid_size = (padded_height // self.patch_size, padded_width // self.patch_size)
        if tuple(self.model.grid_size) != current_grid_size:
            # NOTE: This replaces the position embedding module. That is fine for
            # this project because Cityscapes uses a fixed resolution during training.
            self._set_rectangular_grid(padded_height=padded_height,
                                       padded_width=padded_width)

        outputs = self.model(pixel_values=x)
        prediction_pairs = self._extract_prediction_pairs(outputs)

        if len(prediction_pairs) == 0:
            raise ValueError("EoMT did not return any query predictions")

        dense_logits_sum = None
        for class_queries_logits, masks_queries_logits in prediction_pairs:
            dense_logits = self._query_predictions_to_dense_logits(
                class_queries_logits=class_queries_logits,
                masks_queries_logits=masks_queries_logits,
                padded_height=padded_height,
                padded_width=padded_width,
            )

            if dense_logits_sum is None:
                dense_logits_sum = dense_logits
            else:
                dense_logits_sum = dense_logits_sum + dense_logits

        dense_logits = dense_logits_sum / len(prediction_pairs) #type:ignore
        return dense_logits[..., :original_height, :original_width]
