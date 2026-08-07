"""Temporal and volume-aware pseudo-label bank for PROMISE12.

The bank is anchored to a fixed UniMatch best checkpoint. Historical
checkpoints and Monte-Carlo dropout predictions are used to estimate
reliability; historical predictions never directly replace the anchor target.
"""

import os
import random
import re
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from networks.unet import UNet


def case_from_slice(slice_name):
    return str(slice_name).split("_slice", 1)[0]


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("net", "state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Unsupported checkpoint format")


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = extract_state_dict(checkpoint)
    state = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model = UNet(in_chns=1, class_num=2).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def model_logits(model, images):
    output = model(images)
    return output[0] if isinstance(output, tuple) else output


def set_mc_dropout(model):
    """Enable dropout while keeping batch-normalization layers in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def select_history_checkpoints(primary_path, history_dir, history_count):
    primary_path = os.path.abspath(primary_path)
    if history_count <= 0 or not history_dir or not os.path.isdir(history_dir):
        return [primary_path]

    candidates = []
    for name in os.listdir(history_dir):
        match = re.fullmatch(r"iter_(\d+)\.pth", name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(history_dir, name)))
    candidates.sort()
    if not candidates:
        return [primary_path]

    take = min(int(history_count), len(candidates))
    positions = np.linspace(0, len(candidates) - 1, take)
    chosen = []
    for position in positions:
        path = os.path.abspath(candidates[int(round(float(position)))][1])
        if path != primary_path and path not in chosen:
            chosen.append(path)
    return [primary_path] + chosen


class BankInferenceDataset(Dataset):
    def __init__(self, root_path, sample_names, patch_size):
        self.root_path = os.path.abspath(root_path)
        self.sample_names = list(sample_names)
        self.patch_size = tuple(int(value) for value in patch_size)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        name = self.sample_names[index]
        path = os.path.join(self.root_path, "data", "slices", name + ".h5")
        with h5py.File(path, "r") as handle:
            image = handle["image"][:].astype(np.float32)
        height, width = image.shape
        image = zoom(
            image,
            (self.patch_size[0] / height, self.patch_size[1] / width),
            order=0,
        )
        return torch.from_numpy(image.copy()).unsqueeze(0), name


def mask_boundary(mask, radius=2):
    mask = mask.float().unsqueeze(1)
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(mask, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-mask, kernel, stride=1, padding=radius)
    return (dilated - eroded > 0).squeeze(1)


def prediction_reliability(mean_probability, variance):
    eps = 1e-6
    probability = mean_probability.clamp(eps, 1.0 - eps)
    confidence = (2.0 * (probability - 0.5).abs()).clamp(0.0, 1.0)
    stability = (1.0 - 2.0 * variance.clamp_min(0.0).sqrt()).clamp(0.0, 1.0)
    entropy = -(
        probability * probability.log()
        + (1.0 - probability) * (1.0 - probability).log()
    ) / float(np.log(2.0))
    reliability = confidence * stability * (1.0 - 0.5 * entropy)
    return reliability.clamp(0.0, 1.0), entropy.clamp(0.0, 1.0)


def _case_scores(bank_maps, case_member_areas):
    per_case_fg = defaultdict(list)
    per_case_boundary = defaultdict(list)
    for value in bank_maps.values():
        case = value["case"]
        probability = value["prob"].float()
        reliability = value["reliability"].float()
        foreground = probability >= 0.5
        boundary = value["boundary"].bool()
        if foreground.any():
            per_case_fg[case].append(float(reliability[foreground].mean()))
        else:
            per_case_fg[case].append(float(reliability.mean()))
        if boundary.any():
            per_case_boundary[case].append(float(reliability[boundary].mean()))
        else:
            per_case_boundary[case].append(float(reliability.mean()))

    scores = {}
    for case, areas in case_member_areas.items():
        area_values = np.asarray(areas, dtype=np.float64)
        mean_area = float(area_values.mean())
        area_cv = float(area_values.std() / max(mean_area, 1.0))
        volume_stability = (
            max(0.0, 1.0 - min(area_cv, 1.0))
            if mean_area > 1.0
            else 0.0
        )
        foreground_reliability = float(np.mean(per_case_fg[case]))
        boundary_reliability = float(np.mean(per_case_boundary[case]))
        scores[case] = (
            0.45 * boundary_reliability
            + 0.35 * foreground_reliability
            + 0.20 * volume_stability
        )

    ordered = sorted(scores, key=scores.get, reverse=True)
    denominator = max(len(ordered) - 1, 1)
    percentiles = {
        case: float(rank) / float(denominator)
        for rank, case in enumerate(ordered)
    }
    return scores, percentiles


@torch.no_grad()
def build_temporal_bank(
    root_path,
    sample_names,
    primary_checkpoint,
    history_dir,
    history_count,
    mc_passes,
    patch_size,
    device,
    inference_batch_size=12,
    num_workers=2,
    logger=None,
):
    checkpoint_paths = select_history_checkpoints(
        primary_checkpoint, history_dir, history_count
    )
    if logger:
        logger.info("Temporal bank checkpoints: %s", checkpoint_paths)

    models = [load_checkpoint(path, device) for path in checkpoint_paths]
    dataset = BankInferenceDataset(root_path, sample_names, patch_size)
    loader = DataLoader(
        dataset,
        batch_size=int(inference_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )

    maps = {}
    member_count = len(models) + max(int(mc_passes), 0)
    case_member_areas = defaultdict(lambda: np.zeros(member_count, dtype=np.float64))

    for images, names in loader:
        images = images.to(device, non_blocking=True)
        for model in models:
            model.eval()

        deterministic = [
            torch.softmax(model_logits(model, images), dim=1)[:, 1]
            for model in models
        ]

        anchor_members = [deterministic[0]]
        set_mc_dropout(models[0])
        for _ in range(max(int(mc_passes), 0)):
            anchor_members.append(
                torch.softmax(model_logits(models[0], images), dim=1)[:, 1]
            )

        # The target remains anchored to the fixed best UniMatch checkpoint.
        target_probability = torch.stack(anchor_members, dim=0).mean(dim=0)
        reliability_members = deterministic + anchor_members[1:]
        member_stack = torch.stack(reliability_members, dim=0)
        variance = member_stack.var(dim=0, unbiased=False)
        reliability, entropy = prediction_reliability(
            target_probability, variance
        )
        boundary = mask_boundary(target_probability >= 0.5)

        for batch_index, name in enumerate(names):
            name = str(name)
            case = case_from_slice(name)
            for member_index, member in enumerate(reliability_members):
                case_member_areas[case][member_index] += float(
                    (member[batch_index] >= 0.5).sum().item()
                )
            maps[name] = {
                "case": case,
                "prob": target_probability[batch_index].cpu().half(),
                "reliability": reliability[batch_index].cpu().half(),
                "entropy": entropy[batch_index].cpu().half(),
                "boundary": boundary[batch_index].cpu().to(torch.uint8),
            }

    case_scores, case_percentiles = _case_scores(maps, case_member_areas)
    bank = {
        "version": 1,
        "anchor_checkpoint": os.path.abspath(primary_checkpoint),
        "history_checkpoints": checkpoint_paths[1:],
        "mc_passes": int(mc_passes),
        "maps": maps,
        "case_scores": case_scores,
        "case_percentiles": case_percentiles,
        "refresh_count": 0,
    }
    if logger:
        logger.info(
            "Temporal bank built: slices=%d cases=%d score_range=[%.4f, %.4f]",
            len(maps),
            len(case_scores),
            min(case_scores.values()),
            max(case_scores.values()),
        )
    del models
    torch.cuda.empty_cache()
    return bank


@torch.no_grad()
def refresh_temporal_bank(
    bank,
    model,
    root_path,
    sample_names,
    patch_size,
    device,
    mc_passes=4,
    update_margin=0.02,
    temporal_decay=0.8,
    inference_batch_size=12,
    logger=None,
):
    """Conservatively update a bank using a newer EMA model.

    A pixel is updated only when the candidate is more reliable than the
    stored target. Class-changing updates require near-certain predictions.
    """
    dataset = BankInferenceDataset(root_path, sample_names, patch_size)
    loader = DataLoader(
        dataset,
        batch_size=int(inference_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    was_training = model.training
    accepted_pixels = 0
    total_pixels = 0

    for images, names in loader:
        images = images.to(device, non_blocking=True)
        model.eval()
        candidates = [
            torch.softmax(model_logits(model, images), dim=1)[:, 1]
        ]
        set_mc_dropout(model)
        for _ in range(max(int(mc_passes), 0)):
            candidates.append(
                torch.softmax(model_logits(model, images), dim=1)[:, 1]
            )
        candidate_stack = torch.stack(candidates, dim=0)
        candidate_probability = candidate_stack.mean(dim=0)
        candidate_variance = candidate_stack.var(dim=0, unbiased=False)
        candidate_reliability, candidate_entropy = prediction_reliability(
            candidate_probability, candidate_variance
        )

        for batch_index, name in enumerate(names):
            name = str(name)
            stored = bank["maps"][name]
            old_probability = stored["prob"].to(device=device, dtype=torch.float32)
            old_reliability = stored["reliability"].to(
                device=device, dtype=torch.float32
            )
            new_probability = candidate_probability[batch_index]
            new_reliability = candidate_reliability[batch_index]
            class_agreement = (
                (new_probability >= 0.5) == (old_probability >= 0.5)
            )
            near_certain = torch.maximum(
                new_probability, 1.0 - new_probability
            ) >= 0.99
            accept = (
                new_reliability >= old_reliability + float(update_margin)
            ) & (class_agreement | near_certain)

            blended = (
                float(temporal_decay) * old_probability
                + (1.0 - float(temporal_decay)) * new_probability
            )
            updated_probability = torch.where(accept, blended, old_probability)
            updated_reliability = torch.where(
                accept, new_reliability, old_reliability
            )
            updated_entropy = torch.where(
                accept,
                candidate_entropy[batch_index],
                stored["entropy"].to(device=device, dtype=torch.float32),
            )

            stored["prob"] = updated_probability.cpu().half()
            stored["reliability"] = updated_reliability.cpu().half()
            stored["entropy"] = updated_entropy.cpu().half()
            stored["boundary"] = (
                mask_boundary(updated_probability.unsqueeze(0) >= 0.5)[0]
                .cpu()
                .to(torch.uint8)
            )
            accepted_pixels += int(accept.sum().item())
            total_pixels += int(accept.numel())

    bank["refresh_count"] = int(bank.get("refresh_count", 0)) + 1
    if was_training:
        model.train()
    else:
        model.eval()
    fraction = float(accepted_pixels) / float(max(total_pixels, 1))
    if logger:
        logger.info(
            "Temporal bank refresh %d: accepted_pixels=%d/%d (%.6f)",
            bank["refresh_count"],
            accepted_pixels,
            total_pixels,
            fraction,
        )
    return fraction


def save_temporal_bank(bank, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(bank, path)


def load_temporal_bank(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TemporalBankDataset(Dataset):
    """PROMISE12 training slices with spatially aligned bank targets."""

    def __init__(
        self,
        root_path,
        bank,
        labeled_slice_count,
        patch_size=(256, 256),
    ):
        self.root_path = os.path.abspath(root_path)
        with open(
            os.path.join(self.root_path, "train_slices.list"),
            "r",
            encoding="utf-8",
        ) as handle:
            self.sample_names = [line.strip() for line in handle if line.strip()]
        self.bank = bank
        self.labeled_slice_count = int(labeled_slice_count)
        self.patch_size = tuple(int(value) for value in patch_size)

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def _rot_flip(arrays):
        k = np.random.randint(0, 4)
        axis = np.random.randint(0, 2)
        return [
            np.flip(np.rot90(array, k), axis=axis).copy()
            for array in arrays
        ]

    @staticmethod
    def _rotate(arrays):
        angle = np.random.randint(-20, 20)
        orders = (0, 0, 1, 1, 0)
        return [
            ndimage.rotate(
                array,
                angle,
                order=order,
                reshape=False,
                mode="constant",
                cval=0,
            )
            for array, order in zip(arrays, orders)
        ]

    def __getitem__(self, index):
        name = self.sample_names[index]
        path = os.path.join(self.root_path, "data", "slices", name + ".h5")
        with h5py.File(path, "r") as handle:
            image = handle["image"][:].astype(np.float32)
            label = handle["label"][:].astype(np.uint8)

        if index < self.labeled_slice_count:
            probability = np.zeros_like(image, dtype=np.float32)
            reliability = np.zeros_like(image, dtype=np.float32)
            boundary = np.zeros_like(image, dtype=np.uint8)
            percentile = 0.0
        else:
            value = self.bank["maps"].get(name)
            if value is None:
                raise KeyError("Pseudo-label bank has no entry for {}".format(name))
            probability = value["prob"].float().numpy()
            reliability = value["reliability"].float().numpy()
            boundary = value["boundary"].numpy().astype(np.uint8)
            percentile = float(
                self.bank["case_percentiles"].get(value["case"], 1.0)
            )

        arrays = [image, label, probability, reliability, boundary]
        if random.random() > 0.5:
            arrays = self._rot_flip(arrays)
        elif random.random() > 0.5:
            arrays = self._rotate(arrays)
        image, label, probability, reliability, boundary = arrays

        # Image/label live on the raw H5 grid, while bank maps are stored
        # at the network patch grid. Apply the same spatial transform above,
        # then resize each grid with its own scale so every returned tensor is
        # exactly patch_size. Reusing the image scale for a 256x256 bank map
        # from a 512x512 H5 slice incorrectly produced a 128x128 tensor.
        image_scale = (
            self.patch_size[0] / image.shape[0],
            self.patch_size[1] / image.shape[1],
        )
        bank_scale = (
            self.patch_size[0] / probability.shape[0],
            self.patch_size[1] / probability.shape[1],
        )
        image = zoom(image, image_scale, order=0)
        label = zoom(label, image_scale, order=0)
        probability = np.clip(
            zoom(probability, bank_scale, order=1), 0.0, 1.0
        )
        reliability = np.clip(
            zoom(reliability, bank_scale, order=1), 0.0, 1.0
        )
        boundary = zoom(boundary, bank_scale, order=0)
        expected_shape = self.patch_size
        actual_shapes = {
            "image": image.shape,
            "label": label.shape,
            "probability": probability.shape,
            "reliability": reliability.shape,
            "boundary": boundary.shape,
        }
        if any(shape != expected_shape for shape in actual_shapes.values()):
            raise RuntimeError(
                "Spatial alignment failed for {}: expected {}, got {}".format(
                    name, expected_shape, actual_shapes
                )
            )

        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.uint8)),
            "bank_probability": torch.from_numpy(
                probability.astype(np.float32)
            ),
            "bank_reliability": torch.from_numpy(
                reliability.astype(np.float32)
            ),
            "bank_boundary": torch.from_numpy(
                (boundary > 0).astype(np.float32)
            ),
            "case_percentile": torch.tensor(percentile, dtype=torch.float32),
            "case": name,
        }
