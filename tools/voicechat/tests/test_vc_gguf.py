#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vc_gguf import dequant_q4_0, dequant_q8_0, source_quantization


class TestVoiceChatGGUF(unittest.TestCase):
    def test_dequant_q8_0(self):
        scale = np.array([0.5], dtype=np.float16).view(np.uint8)
        values = np.array([
            -128, -127, -64, -32, -16, -8, -4, -2,
            -1, 0, 1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13, 14,
            15, 16, 24, 32, 48, 64, 96, 127,
        ], dtype=np.int8)
        raw = scale.tobytes() + values.tobytes()
        actual = dequant_q8_0(raw, 32)
        np.testing.assert_array_equal(actual, values.astype(np.float32) * 0.5)

    def test_dequant_q4_0_regression(self):
        scale = np.array([2.0], dtype=np.float16).view(np.uint8)
        packed = np.arange(16, dtype=np.uint8) | (np.arange(15, -1, -1, dtype=np.uint8) << 4)
        actual = dequant_q4_0(scale.tobytes() + packed.tobytes(), 32)
        expected = np.concatenate([
            np.arange(16, dtype=np.float32) - 8,
            np.arange(15, -1, -1, dtype=np.float32) - 8,
        ]) * 2.0
        np.testing.assert_array_equal(actual, expected)

    def test_file_type_metadata(self):
        for value, expected in ((2, "Q4_0"), (7, "Q8_0")):
            src = SimpleNamespace(kv={"general.file_type": value}, tensors={})
            self.assertEqual(source_quantization(src), (expected, False))

    def test_file_type_inference(self):
        q4 = SimpleNamespace(kv={}, tensors={"weight": {"ty": "Q4_0", "elements": 64}})
        q8 = SimpleNamespace(kv={}, tensors={"weight": {"ty": "Q8_0", "elements": 64}})
        self.assertEqual(source_quantization(q4), ("Q4_0", True))
        self.assertEqual(source_quantization(q8), ("Q8_0", True))

    def test_unsupported_file_type(self):
        src = SimpleNamespace(kv={"general.file_type": 1}, tensors={})
        with self.assertRaisesRegex(SystemExit, "unsupported general.file_type"):
            source_quantization(src)


if __name__ == "__main__":
    unittest.main()
