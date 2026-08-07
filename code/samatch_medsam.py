"""LiteMedSAM pieces used by full SAMatch.

Architecture and hyperparameters follow the official SAMatch implementation at
https://github.com/apple1986/SAMatch, commit
0ab023e643177a8a9dc6f76181c92b52225a71eb.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_torch_load(path, map_location="cpu", weights_only=False):
    try:
        return torch.load(
            str(path), map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_module(name, path, package=False):
    kwargs = {}
    if package:
        kwargs["submodule_search_locations"] = [str(path.parent)]
    spec = importlib.util.spec_from_file_location(name, str(path), **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load {} from {}".format(name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MedSAMLite(nn.Module):
    """Official SAMatch LiteMedSAM wrapper with checkpoint-compatible names."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        image_embedding = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=boxes, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions


def build_medsam_lite(checkpoint_path, source_dir, device):
    source_dir = Path(source_dir).resolve()
    modeling_dir = (
        source_dir / "model_sam" / "segment_anything" / "modeling")
    tiny_vit_path = source_dir / "networks" / "tiny_vit_sam.py"
    required = (
        modeling_dir / "__init__.py",
        modeling_dir / "mask_decoder.py",
        modeling_dir / "prompt_encoder.py",
        modeling_dir / "transformer.py",
        tiny_vit_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete official SAMatch source: {}".format(", ".join(missing)))

    modeling = load_module(
        "_samatch_official_modeling",
        modeling_dir / "__init__.py",
        package=True,
    )
    tiny_vit = load_module("_samatch_official_tiny_vit", tiny_vit_path)

    image_encoder = tiny_vit.TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = modeling.PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = modeling.MaskDecoder(
        num_multimask_outputs=3,
        transformer=modeling.TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    model = MedSAMLite(image_encoder, mask_decoder, prompt_encoder)
    checkpoint = safe_torch_load(
        checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.to(device)
    logging.info(
        "Loaded LiteMedSAM from %s (%d parameters)",
        checkpoint_path,
        sum(parameter.numel() for parameter in model.parameters()))
    return model


def masks_to_boxes(masks, shift):
    """Convert binary BxHxW masks to Bx1x4 boxes and a valid-prompt mask."""
    batch, height, width = masks.shape
    boxes = torch.zeros(
        (batch, 1, 4), dtype=torch.float32, device=masks.device)
    valid = torch.zeros(batch, dtype=torch.bool, device=masks.device)
    for index in range(batch):
        coordinates = torch.nonzero(masks[index] > 0, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        y_min = max(0, int(coordinates[:, 0].min().item()) - shift)
        y_max = min(height - 1, int(coordinates[:, 0].max().item()) + shift)
        x_min = max(0, int(coordinates[:, 1].min().item()) - shift)
        x_max = min(width - 1, int(coordinates[:, 1].max().item()) + shift)
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes[index, 0] = torch.tensor(
            [x_min, y_min, x_max, y_max],
            dtype=torch.float32,
            device=masks.device,
        )
        valid[index] = True
    return boxes, valid


def binary_iou(prediction, target):
    prediction = prediction.bool()
    target = target.bool()
    intersection = torch.logical_and(
        prediction, target).flatten(1).sum(1)
    union = torch.logical_or(prediction, target).flatten(1).sum(1)
    return (
        intersection.float() /
        union.float().clamp_min(1.0)).unsqueeze(1)


def binary_dice_loss(logits, target):
    probability = torch.sigmoid(logits)
    target = target.float()
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dimensions)
    denominator = (
        probability.square().sum(dimensions) +
        target.square().sum(dimensions))
    return (
        1.0 -
        (2.0 * intersection + 1e-5) /
        (denominator + 1e-5)).mean()


def sam_mask_loss(logits, target):
    target = target.float()
    return (
        binary_dice_loss(logits, target) +
        F.binary_cross_entropy_with_logits(logits, target))
