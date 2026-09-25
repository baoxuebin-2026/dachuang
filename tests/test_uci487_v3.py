"""Check the two independent blocking axes and exclusion of target/clock features."""

import unittest

import numpy as np
import pandas as pd

from scripts.run_uci487_v3 import MODELS, TARGET, make_manifest


class BlockedProtocolTest(unittest.TestCase):
    def test_day_and_position_are_both_disjoint(self):
        first = pd.DataFrame({"segment": range(100), TARGET: np.repeat(np.arange(10), 10)})
        days = [f"day-{i:02d}" for i in range(13)]
        manifest = make_manifest(days, first)
        train = manifest[manifest.split == "train"]
        validation = manifest[manifest.split == "validation"]
        test = manifest[manifest.split == "test"]
        self.assertEqual((len(train), len(validation), len(test)), (480, 40, 60))
        for a, b in ((train, validation), (train, test), (validation, test)):
            self.assertTrue(set(a.day).isdisjoint(b.day))
            self.assertTrue(set(a.segment).isdisjoint(b.segment))
        for part, n in ((train, 48), (validation, 4), (test, 6)):
            self.assertTrue((part.co_level_offline_ppm.value_counts() == n).all())
        self.assertEqual(manifest.loc[manifest.split == "excluded_cross_cell"].shape[0], 720)

    def test_inputs_exclude_reference_and_program_markers(self):
        forbidden = {TARGET, "segment", "day", "Time (s)", "CO (ppm)",
                     "co_level_offline_ppm", "Flow rate (mL/min)"}
        for name, columns in MODELS.items():
            with self.subTest(model=name):
                self.assertFalse(forbidden.intersection(columns))


if __name__ == "__main__":
    unittest.main()
