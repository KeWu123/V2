"""Validate every required PROMISE12 H5 and rebuild pointer files if needed."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from promise12_protocol import validate_data_root


HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ARCHIVE_MD5 = "f6d23994117c989daf07e5291edd0aea"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    return parser.parse_args()


def list_values(path):
    if not path.is_file():
        return []
    return [
        line.strip().split(".")[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def required_h5_paths(data_root):
    train_slices = list_values(data_root / "train_slices.list")
    val_cases = list_values(data_root / "val.list")
    test_cases = list_values(data_root / "test.list")
    if not train_slices or not val_cases or not test_cases:
        return []
    paths = [
        data_root / "data" / "slices" / f"{name}.h5"
        for name in train_slices
    ]
    paths.extend(
        data_root / "data" / f"{case}.h5"
        for case in val_cases + test_cases
    )
    return paths


def first_invalid(data_root):
    split = validate_data_root(data_root, require_h5=False)
    if split["errors"]:
        return data_root / "<invalid 35-5-10 split>"
    paths = required_h5_paths(data_root)
    if not paths:
        return data_root / "<missing dataset lists>"
    for path in paths:
        if not path.is_file() or path.stat().st_size < 1024:
            return path
        with path.open("rb") as handle:
            if handle.read(8) != HDF5_MAGIC:
                return path
    return None


def verify_archive(archive):
    if not archive.is_file():
        raise SystemExit(f"Missing PROMISE12 archive: {archive}")
    digest = hashlib.md5()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != ARCHIVE_MD5:
        raise SystemExit(
            f"PROMISE12 archive MD5 mismatch: expected {ARCHIVE_MD5}, "
            f"got {actual}")


def extract_archive(archive, raw_root):
    raw_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise SystemExit(f"Corrupt ZIP member: {corrupt}")
        root = raw_root.resolve()
        for member in bundle.infolist():
            destination = (root / member.filename).resolve()
            if os.path.commonpath((str(root), str(destination))) != str(root):
                raise SystemExit(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(root)
    case00 = raw_root / "Case00.raw"
    if not case00.is_file() or case00.stat().st_size < 1_000_000:
        raise SystemExit(f"Extraction did not produce real raw data: {case00}")


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    invalid = first_invalid(data_root)
    if invalid is None:
        count = len(required_h5_paths(data_root))
        print(f"PROMISE12 H5 signature check passed: {count} required files")
        return

    print(f"Invalid/placeholder PROMISE12 H5 detected: {invalid}")
    print("Rebuilding H5 data from the bundled verified raw archive...")
    verify_archive(args.archive.resolve())
    extract_archive(args.archive.resolve(), args.raw_root.resolve())
    subprocess.run(
        [
            sys.executable,
            str(args.converter.resolve()),
            "--raw_root", str(args.raw_root.resolve()),
            "--out_root", str(data_root),
        ],
        check=True,
    )
    invalid = first_invalid(data_root)
    if invalid is not None:
        raise SystemExit(f"H5 rebuild left an invalid required file: {invalid}")
    count = len(required_h5_paths(data_root))
    print(f"PROMISE12 H5 rebuild and signature check passed: {count} files")


if __name__ == "__main__":
    main()
