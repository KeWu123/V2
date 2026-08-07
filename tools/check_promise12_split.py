"""Fail fast unless a runtime dataset uses the canonical 35/5/10 split."""

import argparse
from pathlib import Path

from promise12_protocol import LABELNUM, validate_data_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--require_h5", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = validate_data_root(args.data_root.resolve(), args.require_h5)
    if result["errors"]:
        details = "\n  - ".join(result["errors"])
        raise SystemExit(
            "PROMISE12 split validation failed. Expected canonical 35/5/10:\n"
            f"  - {details}"
        )
    print(
        "PROMISE12 split OK: "
        f"train={len(result['train_cases'])} "
        f"val={len(result['val_cases'])} "
        f"test={len(result['test_cases'])} "
        f"train_slices={len(result['train_slices'])} "
        f"labelnum={LABELNUM} labeled_slices={result['labeled_slices']}"
    )


if __name__ == "__main__":
    main()
