import argparse
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk

from promise12_protocol import TEST_CASES, TRAIN_CASES, VAL_CASES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "PROMISE12" / "extracted" / "training_data"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "PROMISE12_h5"


def normalize_volume(image):
    image = image.astype(np.float32)
    foreground = image[image > 0]
    if foreground.size == 0:
        foreground = image.reshape(-1)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        std = 1.0
    return ((image - mean) / std).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PROMISE12 volumes to leakage-free H5 files."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Folder containing CaseXX.mhd and CaseXX_segmentation.mhd files.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Destination PROMISE12_h5 folder.",
    )
    return parser.parse_args()


def read_case(case_id, raw_root):
    case = f"Case{case_id:02d}"
    image_itk = sitk.ReadImage(str(raw_root / f"{case}.mhd"))
    label_itk = sitk.ReadImage(str(raw_root / f"{case}_segmentation.mhd"))
    image = normalize_volume(sitk.GetArrayFromImage(image_itk))
    label = (sitk.GetArrayFromImage(label_itk) > 0).astype(np.uint8)
    spacing_zyx = tuple(reversed(image_itk.GetSpacing()))
    return case, image, label, spacing_zyx


def write_h5(path, image, label, spacing):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=image, compression="gzip")
        handle.create_dataset("label", data=label, compression="gzip")
        handle.attrs["spacing"] = np.asarray(spacing, dtype=np.float32)


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw PROMISE12 folder: {raw_root}")
    slices_dir = out_root / "data" / "slices"
    volumes_dir = out_root / "data"
    slices_dir.mkdir(parents=True, exist_ok=True)
    volumes_dir.mkdir(parents=True, exist_ok=True)

    train_cases = list(TRAIN_CASES)
    val_cases = list(VAL_CASES)
    test_cases = list(TEST_CASES)
    train_slices = []
    summary = []

    for requested_case in train_cases + val_cases + test_cases:
        case_id = int(requested_case.replace("Case", ""))
        case, image, label, spacing_zyx = read_case(case_id, raw_root)
        write_h5(volumes_dir / f"{case}.h5", image, label, spacing_zyx)
        summary.append((case, image.shape, int(label.sum()), spacing_zyx))
        if case in train_cases:
            for z_index in range(image.shape[0]):
                slice_name = f"{case}_slice{z_index:03d}"
                train_slices.append(slice_name)
                write_h5(
                    slices_dir / f"{slice_name}.h5",
                    image[z_index],
                    label[z_index],
                    spacing_zyx[1:],
                )

    (out_root / "train_slices.list").write_text("\n".join(train_slices) + "\n", encoding="utf-8")
    (out_root / "train.list").write_text("\n".join(train_cases) + "\n", encoding="utf-8")
    (out_root / "val.list").write_text("\n".join(val_cases) + "\n", encoding="utf-8")
    (out_root / "test.list").write_text("\n".join(test_cases) + "\n", encoding="utf-8")
    with (out_root / "conversion_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"source={raw_root}\n")
        handle.write(f"train_cases={train_cases}\n")
        handle.write(f"val_cases={val_cases}\n")
        handle.write(f"test_cases={test_cases}\n")
        handle.write(f"train_slices={len(train_slices)}\n")
        handle.write("split_protocol=canonical PROMISE12 35/5/10, labelnum=7\n")
        handle.write("train_slices.list contains every slice; no hidden-label filtering is used.\n")
        for case, shape, label_sum, spacing in summary:
            handle.write(f"{case}: shape={shape}, spacing_zyx={spacing}, label_voxels={label_sum}\n")
    print(f"wrote {out_root}")
    print(f"train cases={len(train_cases)} val cases={len(val_cases)} test cases={len(test_cases)}")
    print(f"train slices={len(train_slices)}")


if __name__ == "__main__":
    main()
