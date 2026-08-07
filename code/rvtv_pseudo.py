"""Reference-verified temporal/volume pseudo-label routing for PROMISE12.

The module is deliberately independent of the segmentation architecture.  It
uses a frozen pre-training encoder and ground-truth reference pixels to verify
EMA pseudo labels, while an online per-slice bank measures temporal area
stability and adjacent-slice volume consistency.
"""

import math
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


_SLICE_PATTERN = re.compile(r"^(.*)_slice(\d+)$")


def split_slice_key(key):
    match = _SLICE_PATTERN.match(str(key))
    if match is None:
        return str(key), 0
    return match.group(1), int(match.group(2))


def case_slice_extents(sample_list):
    extents = defaultdict(int)
    for key in sample_list:
        case, index = split_slice_key(key)
        extents[case] = max(extents[case], index)
    return dict(extents)


def slice_zone(key, extents):
    case, index = split_slice_key(key)
    maximum = max(1, int(extents.get(case, index)))
    relative = float(index) / float(maximum)
    return min(2, int(relative * 3.0))


class ReferenceFeatureBank:
    """Frozen labeled feature memory used only to verify pseudo labels."""

    def __init__(self, model, sample_list, feature_level=2,
                 samples_per_slice=32, max_samples=512, topk=5,
                 temperature=0.10):
        self.model = model
        self.extents = case_slice_extents(sample_list)
        self.feature_level = int(feature_level)
        self.samples_per_slice = int(samples_per_slice)
        self.max_samples = int(max_samples)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.features = {}
        self.sources = {}
        self.thresholds = {0: 0.90, 1: 0.90}
        self.calibration_metrics = {}

    @torch.no_grad()
    def _features(self, images):
        encoded = self.model.encoder(images)[self.feature_level]
        return F.normalize(encoded, dim=1)

    @torch.no_grad()
    def build(self, loader, device):
        feature_parts = defaultdict(list)
        source_parts = defaultdict(list)
        self.model.eval()

        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            cases = [str(item) for item in batch['case']]
            features = self._features(images)
            reduced_labels = F.interpolate(
                labels.unsqueeze(1).float(), size=features.shape[-2:],
                mode='nearest').squeeze(1).long()

            for sample_index, case_key in enumerate(cases):
                zone = slice_zone(case_key, self.extents)
                flattened = features[sample_index].permute(1, 2, 0).reshape(
                    -1, features.shape[1])
                flattened_labels = reduced_labels[sample_index].reshape(-1)
                for class_id in (0, 1):
                    candidates = torch.nonzero(
                        flattened_labels == class_id, as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        continue
                    count = min(self.samples_per_slice, candidates.numel())
                    choice = candidates[torch.randperm(
                        candidates.numel(), device=device)[:count]]
                    selected = flattened[choice].detach()
                    for bank_zone in (zone, -1):
                        bank_key = (bank_zone, class_id)
                        feature_parts[bank_key].append(selected)
                        source_parts[bank_key].extend([case_key] * count)

        for bank_key, parts in feature_parts.items():
            combined = torch.cat(parts, dim=0)
            sources = source_parts[bank_key]
            if combined.shape[0] > self.max_samples:
                choice = torch.randperm(
                    combined.shape[0], device=combined.device)[:self.max_samples]
                combined = combined[choice]
                source_indices = choice.detach().cpu().tolist()
                sources = [sources[index] for index in source_indices]
            self.features[bank_key] = F.normalize(combined, dim=1)
            self.sources[bank_key] = sources

        for class_id in (0, 1):
            if (-1, class_id) not in self.features:
                raise RuntimeError(
                    "Reference bank has no samples for class {}".format(class_id))

    def _bank(self, zone, class_id, excluded_case=None):
        bank_key = (zone, class_id)
        if bank_key not in self.features:
            bank_key = (-1, class_id)
        features = self.features[bank_key]
        if excluded_case is None:
            return features
        keep = [
            index for index, source in enumerate(self.sources[bank_key])
            if split_slice_key(source)[0] != split_slice_key(excluded_case)[0]
        ]
        if not keep:
            return self.features[(-1, class_id)]
        keep_tensor = torch.as_tensor(
            keep, dtype=torch.long, device=features.device)
        return features.index_select(0, keep_tensor)

    @torch.no_grad()
    def probability(self, images, cases, exclude_own_case=False,
                    output_size=None):
        features = self._features(images)
        batch_probabilities = []
        for sample_index, case_key in enumerate(cases):
            zone = slice_zone(case_key, self.extents)
            query = features[sample_index].permute(1, 2, 0).reshape(
                -1, features.shape[1])
            class_scores = []
            for class_id in (0, 1):
                bank = self._bank(
                    zone, class_id,
                    case_key if exclude_own_case else None)
                similarities = torch.matmul(query, bank.t())
                k = min(self.topk, similarities.shape[1])
                class_scores.append(
                    similarities.topk(k, dim=1).values.mean(dim=1))
            logits = torch.stack(class_scores, dim=1) / self.temperature
            probability = torch.softmax(logits, dim=1)
            probability = probability.reshape(
                features.shape[-2], features.shape[-1], 2).permute(2, 0, 1)
            batch_probabilities.append(probability)

        probabilities = torch.stack(batch_probabilities, dim=0)
        if output_size is not None and tuple(probabilities.shape[-2:]) != tuple(output_size):
            probabilities = F.interpolate(
                probabilities, size=output_size, mode='bilinear',
                align_corners=False)
        return probabilities

    @staticmethod
    def _precision_threshold(confidence, correct, target_precision):
        if confidence.size == 0:
            return 0.95
        chosen = None
        for threshold in np.linspace(0.50, 0.995, 100):
            accepted = confidence >= threshold
            if accepted.sum() < 256:
                continue
            precision = float(correct[accepted].mean())
            if precision >= target_precision:
                chosen = float(threshold)
                break
        if chosen is None:
            # If the requested precision is not demonstrated on labeled
            # leave-one-case-out queries, do not relax the reference verifier.
            chosen = 0.995
        return chosen

    @torch.no_grad()
    def calibrate(self, loader, device, target_precision=0.95):
        confidence_by_class = {0: [], 1: []}
        correctness_by_class = {0: [], 1: []}
        self.model.eval()
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            cases = [str(item) for item in batch['case']]
            probabilities = self.probability(
                images, cases, exclude_own_case=True, output_size=None)
            reduced_labels = F.interpolate(
                labels.unsqueeze(1).float(), size=probabilities.shape[-2:],
                mode='nearest').squeeze(1).long()
            confidence, prediction = probabilities.max(dim=1)
            for class_id in (0, 1):
                selected = prediction == class_id
                if not selected.any():
                    continue
                confidence_by_class[class_id].append(
                    confidence[selected].detach().cpu().numpy())
                correctness_by_class[class_id].append(
                    prediction[selected].eq(reduced_labels[selected])
                    .float().detach().cpu().numpy())

        for class_id in (0, 1):
            confidence = np.concatenate(confidence_by_class[class_id]) \
                if confidence_by_class[class_id] else np.empty(0)
            correct = np.concatenate(correctness_by_class[class_id]) \
                if correctness_by_class[class_id] else np.empty(0)
            self.thresholds[class_id] = self._precision_threshold(
                confidence, correct, target_precision)
            accepted = confidence >= self.thresholds[class_id]
            precision = float(correct[accepted].mean()) \
                if accepted.any() else 0.0
            coverage = float(accepted.mean()) if accepted.size else 0.0
            self.calibration_metrics[class_id] = {
                'precision': precision,
                'coverage': coverage,
                'samples': int(accepted.sum()),
            }
        return dict(self.thresholds)


class TemporalVolumeBank:
    """Rotation-invariant temporal and adjacent-slice reliability bank."""

    def __init__(self, decay=0.90):
        self.decay = float(decay)
        self.states = {}

    def _neighbor_areas(self, case, index):
        values = []
        for offset in (-2, -1, 1, 2):
            state = self.states.get((case, index + offset))
            if state is not None and state['count'] >= 1:
                values.append(state['mean'])
        return values

    @torch.no_grad()
    def update_and_score(self, cases, pseudo_labels):
        device = pseudo_labels.device
        areas = (pseudo_labels == 1).float().mean(dim=(-2, -1)).cpu().tolist()
        temporal_scores = []
        volume_scores = []

        for case_key, area in zip(cases, areas):
            case, index = split_slice_key(case_key)
            state_key = (case, index)
            state = self.states.get(state_key)
            if state is None or state['count'] < 2:
                temporal_score = 0.0
            else:
                deviation = abs(area - state['mean'])
                scale = math.sqrt(max(state['var'], 0.0)) + 0.02
                temporal_score = math.exp(-deviation / scale)

            neighbors = self._neighbor_areas(case, index)
            if neighbors:
                neighbor_mean = float(sum(neighbors) / len(neighbors))
                volume_score = math.exp(
                    -abs(area - neighbor_mean) /
                    (0.03 + area + neighbor_mean))
            else:
                volume_score = 0.5

            if state is None:
                self.states[state_key] = {
                    'mean': float(area), 'var': 0.0, 'count': 1}
            else:
                delta = float(area) - state['mean']
                state['mean'] = (
                    self.decay * state['mean'] +
                    (1.0 - self.decay) * float(area))
                state['var'] = (
                    self.decay * state['var'] +
                    (1.0 - self.decay) * delta * delta)
                state['count'] += 1

            temporal_scores.append(temporal_score)
            volume_scores.append(volume_score)

        return (
            torch.tensor(temporal_scores, device=device, dtype=torch.float32),
            torch.tensor(volume_scores, device=device, dtype=torch.float32),
        )


def route_pseudo_labels(teacher_probability, teacher_labels,
                        reference_probability, reference_thresholds,
                        temporal_scores, volume_scores,
                        teacher_threshold=0.95,
                        soft_teacher_threshold=0.70,
                        soft_reference_threshold=0.70,
                        temporal_threshold=0.25,
                        volume_threshold=0.25,
                        reference_mix=0.35):
    teacher_confidence = teacher_probability.gather(
        1, teacher_labels.unsqueeze(1)).squeeze(1)
    reference_confidence, reference_labels = reference_probability.max(dim=1)
    threshold_background = teacher_probability.new_tensor(
        float(reference_thresholds[0]))
    threshold_foreground = teacher_probability.new_tensor(
        float(reference_thresholds[1]))
    reference_required = torch.where(
        reference_labels == 1, threshold_foreground, threshold_background)

    structural = (
        (temporal_scores >= temporal_threshold) &
        (volume_scores >= volume_threshold)
    ).unsqueeze(-1).unsqueeze(-1)
    agreement = teacher_labels.eq(reference_labels)
    hard_valid = (
        agreement &
        (teacher_confidence >= teacher_threshold) &
        (reference_confidence >= reference_required) &
        structural
    )

    soft_target = (
        (1.0 - reference_mix) * teacher_probability +
        reference_mix * reference_probability
    )
    soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    disagreement_or_uncertain = (
        (~agreement) |
        (teacher_confidence < teacher_threshold) |
        (reference_confidence < reference_required)
    )
    soft_structural = (
        (temporal_scores >= temporal_threshold * 0.5) &
        (volume_scores >= volume_threshold * 0.5)
    ).unsqueeze(-1).unsqueeze(-1)
    soft_valid = (
        (~hard_valid) & disagreement_or_uncertain &
        (teacher_confidence >= soft_teacher_threshold) &
        (reference_confidence >= soft_reference_threshold) &
        soft_structural
    )

    return {
        'hard_target': teacher_labels,
        'hard_valid': hard_valid,
        'soft_target': soft_target.detach(),
        'soft_valid': soft_valid,
        'teacher_confidence': teacher_confidence,
        'reference_confidence': reference_confidence,
        'reference_labels': reference_labels,
        'agreement': agreement,
    }


def routed_pseudo_loss(logits, route, dice_loss, soft_weight=0.25):
    hard_valid = route['hard_valid']
    hard_float = hard_valid.float()
    per_pixel_ce = F.cross_entropy(
        logits, route['hard_target'].long(), reduction='none')
    hard_ce = (per_pixel_ce * hard_float).sum() / hard_float.sum().clamp_min(1.0)
    hard_dice = dice_loss(
        torch.softmax(logits, dim=1),
        route['hard_target'].unsqueeze(1),
        mask=hard_float.unsqueeze(1))
    hard_loss = 0.5 * (hard_ce + hard_dice)

    soft_valid = route['soft_valid']
    soft_float = soft_valid.float()
    per_pixel_kl = F.kl_div(
        torch.log_softmax(logits, dim=1), route['soft_target'],
        reduction='none').sum(dim=1)
    soft_loss = (
        (per_pixel_kl * soft_float).sum() /
        soft_float.sum().clamp_min(1.0)
    )
    total = hard_loss + float(soft_weight) * soft_loss
    return total, hard_loss, soft_loss
