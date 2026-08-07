"""PROMISE12 three-slice (2.5D) training dataset.

Only the input representation differs from ``dataset.py``: every sample is
``[z-1, z, z+1]`` and the target remains the centre-slice label.  At a volume
boundary the nearest available slice is repeated.  All three channels receive
the same spatial augmentation.
"""

import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def _case_name(sample_name):
    if "_slice" not in sample_name:
        raise ValueError("Invalid PROMISE12 slice name: {}".format(sample_name))
    return sample_name.rsplit("_slice", 1)[0]


class PROMISE12DataSets25D(Dataset):
    """Load adjacent axial slices while preserving train_slices.list order."""

    def __init__(self, base_dir, transform=None, num=None):
        self._base_dir = base_dir
        self.transform = transform
        list_path = os.path.join(base_dir, "train_slices.list")
        with open(list_path, "r") as handle:
            self.sample_list = [line.strip() for line in handle if line.strip()]
        if num is not None:
            self.sample_list = self.sample_list[:num]

        case_slices = {}
        for name in self.sample_list:
            case_slices.setdefault(_case_name(name), []).append(name)

        self.neighbours = {}
        for names in case_slices.values():
            for index, name in enumerate(names):
                self.neighbours[name] = (
                    names[max(index - 1, 0)],
                    name,
                    names[min(index + 1, len(names) - 1)],
                )
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def _read_slice(self, name, with_label=False):
        path = os.path.join(self._base_dir, "data", "slices", name + ".h5")
        with h5py.File(path, "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:] if with_label else None
        return image, label

    def __getitem__(self, index):
        name = self.sample_list[index]
        neighbour_names = self.neighbours[name]
        images = []
        centre_label = None
        for channel, neighbour_name in enumerate(neighbour_names):
            image, label = self._read_slice(
                neighbour_name, with_label=(channel == 1))
            images.append(image)
            if channel == 1:
                centre_label = label

        sample = {
            "image": np.stack(images, axis=0),
            "label": centre_label,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample["case"] = name
        return sample


def _random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(0, 1))
    spatial_axis = np.random.randint(0, 2)
    image = np.flip(image, axis=spatial_axis + 1).copy()
    label = np.flip(label, axis=spatial_axis).copy()
    return image, label


def _random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(
        image, angle, axes=(1, 2), order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator25D:
    """The original UniMatch spatial transform, shared by three channels."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = _random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = _random_rotate(image, label)

        height, width = image.shape[1:]
        scale = (
            1,
            self.output_size[0] / height,
            self.output_size[1] / width,
        )
        image = zoom(image, scale, order=0)
        label = zoom(
            label,
            (self.output_size[0] / height, self.output_size[1] / width),
            order=0,
        )
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.uint8)),
        }
