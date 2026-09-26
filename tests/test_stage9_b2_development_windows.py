"""Boundary tests for causal proxy events and split-safe minute labels."""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.build_stage9_b2_development_windows import CONFIG, audit_channel


CFG = json.loads(Path(CONFIG).read_text(encoding="utf-8"))


def run(values, stamps=None):
    if stamps is None:
        stamps = np.arange(len(values), dtype=np.int64)
    return audit_channel(stamps, (stamps % 60).astype(np.int8), values, CFG)


class ProtocolBoundaryTests(unittest.TestCase):
    def test_horizon_border_and_full_history(self):
        x = np.full(1600, 0.4, dtype=np.float32)
        x[1020:1050] = 1.1
        counts, events, windows = run(x)
        labels = {row["source_row"]: row["proxy_label"] for row in windows}
        self.assertEqual((counts["clean_sustained_proxy_events"],
                          counts["evaluable_proxy_events"]), (1, 1))
        self.assertEqual(events[0]["source_row"], 1020)
        self.assertNotIn(599, labels)  # 600 prior seconds require row -1.
        self.assertEqual(labels[659], 0)  # 1020 is 361 s ahead.
        self.assertEqual(labels[719], 1)  # 1020 is 301 s ahead.
        self.assertNotIn(1079, labels)  # Recently crossed, cannot count early.

    def test_short_peak_not_an_event_and_no_180s_quiet(self):
        x = np.full(1700, 0.4, dtype=np.float32)
        x[800:810] = 1.1
        x[950:980] = 1.1  # 140 s quiet: excluded, despite 30 s crossing.
        x[1200:1230] = 1.1  # 220 s quiet: retained.
        counts, events, _ = run(x)
        self.assertEqual(counts["runs_at_least_30_seconds"], 2)
        self.assertEqual([row["source_row"] for row in events], [1200])

    def test_gap_blocks_future_window_and_breaks_high_run(self):
        x = np.full(1600, 0.4, dtype=np.float32)
        x[1000:1030] = 1.1
        stamps = np.arange(1600, dtype=np.int64)
        stamps[1010:] += 3600
        counts, events, windows = run(x, stamps)
        self.assertEqual(counts["timestamp_discontinuities"], 1)
        self.assertEqual(counts["clean_sustained_proxy_events"], 0)
        self.assertGreater(counts["anchor_status"].get("invalid_future", 0), 0)
        self.assertFalse(any(row["source_row"] == 699 and row["proxy_label"]
                             for row in windows))

    def test_invalid_values_are_unknown_not_normal(self):
        x = np.full(1600, 0.4, dtype=np.float32)
        x[1000:1030] = 1.1
        x[1010] = 31.0
        counts, events, windows = run(x)
        self.assertEqual(events, [])
        self.assertEqual(counts["invalid_target_seconds"], 1)
        self.assertGreater(counts["anchor_status"].get("invalid_future", 0), 0)
        self.assertFalse(any(row["source_row"] == 699 for row in windows))


if __name__ == "__main__":
    unittest.main()
