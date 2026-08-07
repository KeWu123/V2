import tempfile
import unittest
from pathlib import Path
import sys


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from promise12_protocol import (  # noqa: E402
    TEST_CASES,
    TRAIN_CASES,
    VAL_CASES,
    validate_data_root,
    validate_partition,
)


class Promise12ProtocolTest(unittest.TestCase):
    def test_canonical_partition_covers_all_50_cases(self):
        self.assertEqual(validate_partition(TRAIN_CASES, VAL_CASES, TEST_CASES), [])
        self.assertEqual(len(TRAIN_CASES), 35)
        self.assertEqual(len(VAL_CASES), 5)
        self.assertEqual(len(TEST_CASES), 10)
        self.assertEqual(len(set(TRAIN_CASES + VAL_CASES + TEST_CASES)), 50)

    def test_runtime_lists_and_slice_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.list").write_text(
                "\n".join(TRAIN_CASES) + "\n", encoding="utf-8"
            )
            (root / "val.list").write_text(
                "\n".join(VAL_CASES) + "\n", encoding="utf-8"
            )
            (root / "test.list").write_text(
                "\n".join(TEST_CASES) + "\n", encoding="utf-8"
            )
            slices = [f"{case}_slice000" for case in TRAIN_CASES]
            (root / "train_slices.list").write_text(
                "\n".join(slices) + "\n", encoding="utf-8"
            )
            result = validate_data_root(root)
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["labeled_slices"], 7)

    def test_legacy_42_4_4_partition_is_rejected(self):
        train = tuple(f"Case{index:02d}" for index in range(42))
        val = tuple(f"Case{index:02d}" for index in range(42, 46))
        test = tuple(f"Case{index:02d}" for index in range(46, 50))
        self.assertTrue(validate_partition(train, val, test))


if __name__ == "__main__":
    unittest.main()
