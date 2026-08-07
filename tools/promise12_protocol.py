"""Canonical PROMISE12 split and lightweight runtime validation."""

from pathlib import Path
import re


TRAIN_CASES = (
    "Case48", "Case35", "Case04", "Case25", "Case23", "Case15", "Case08",
    "Case00", "Case17", "Case44", "Case47", "Case11", "Case18", "Case26",
    "Case42", "Case33", "Case24", "Case14", "Case29", "Case06", "Case27",
    "Case41", "Case28", "Case13", "Case37", "Case12", "Case40", "Case20",
    "Case01", "Case32", "Case19", "Case21", "Case39", "Case10", "Case03",
)
VAL_CASES = ("Case31", "Case02", "Case07", "Case46", "Case22")
TEST_CASES = (
    "Case09", "Case30", "Case45", "Case34", "Case43",
    "Case36", "Case38", "Case16", "Case05", "Case49",
)
LABELNUM = 7

_SLICE_PATTERN = re.compile(r"^(Case\d{2})_slice_?\d+$")


def read_list(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [
        line.strip().split(".")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def slice_case(name):
    match = _SLICE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid PROMISE12 slice name: {name}")
    return match.group(1)


def validate_partition(train_cases, val_cases, test_cases):
    errors = []
    actual = (tuple(train_cases), tuple(val_cases), tuple(test_cases))
    expected = (TRAIN_CASES, VAL_CASES, TEST_CASES)
    labels = ("train", "val", "test")
    for label, values, wanted in zip(labels, actual, expected):
        if values != wanted:
            errors.append(
                f"{label}.list does not match the canonical order "
                f"({len(values)} found, {len(wanted)} expected)"
            )

    combined = list(train_cases) + list(val_cases) + list(test_cases)
    expected_all = {f"Case{index:02d}" for index in range(50)}
    if len(combined) != len(set(combined)):
        errors.append("train/val/test contain duplicate cases")
    if set(combined) != expected_all:
        missing = sorted(expected_all.difference(combined))
        extra = sorted(set(combined).difference(expected_all))
        errors.append(f"partition coverage mismatch: missing={missing}, extra={extra}")
    return errors


def validate_data_root(data_root, require_h5=False):
    data_root = Path(data_root)
    train_cases = read_list(data_root / "train.list")
    val_cases = read_list(data_root / "val.list")
    test_cases = read_list(data_root / "test.list")
    train_slices = read_list(data_root / "train_slices.list")
    errors = validate_partition(train_cases, val_cases, test_cases)

    if not train_slices:
        errors.append("train_slices.list is missing or empty")
        slice_cases = []
    else:
        try:
            slice_cases = [slice_case(name) for name in train_slices]
        except ValueError as exc:
            errors.append(str(exc))
            slice_cases = []

    if slice_cases:
        transitions = []
        for case in slice_cases:
            if not transitions or transitions[-1] != case:
                transitions.append(case)
        if transitions != train_cases:
            errors.append(
                "train_slices.list is not grouped in the exact train.list order"
            )
        non_train = sorted(set(slice_cases).difference(train_cases))
        if non_train:
            errors.append(f"train_slices.list contains non-train cases: {non_train}")

    if require_h5 and train_slices:
        for name in train_slices:
            if not (data_root / "data" / "slices" / f"{name}.h5").is_file():
                errors.append(f"missing slice H5: {name}.h5")
                break
        for case in val_cases + test_cases:
            if not (data_root / "data" / f"{case}.h5").is_file():
                errors.append(f"missing volume H5: {case}.h5")
                break

    labeled_cases = set(train_cases[:LABELNUM])
    labeled_slices = sum(case in labeled_cases for case in slice_cases)
    return {
        "errors": errors,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "train_slices": train_slices,
        "labeled_slices": labeled_slices,
    }
