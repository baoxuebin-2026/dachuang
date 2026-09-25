"""Protect Stage 7A from segment, release-clock, and denominator mistakes."""

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from scripts.run_stage7_a import eligible_onsets, sb112_day, timeslice_mask, transitions


class Stage7Boundaries(unittest.TestCase):
    def test_gap_and_pre_block_history_do_not_supply_positive_event(self):
        runs = [
            {"state": "both_zero", "start_s": 90, "end_s": 160, "duration_s": 70},
            {"state": "target_present", "start_s": 160, "end_s": 230, "duration_s": 70},
            {"state": "both_zero", "start_s": 230, "end_s": 300, "duration_s": 70},
            {"state": "target_present", "start_s": 300, "end_s": 400, "duration_s": 100},
        ]
        # The 160 s onset must be excluded: its required background began before block 100.
        self.assertEqual(eligible_onsets(runs, (100, 400), 60, 60, 60), [300])
        self.assertFalse(timeslice_mask(runs, (100, 400), 120, 450).any())

    def test_alarms_require_causal_persistence_and_reset_at_block(self):
        score = np.zeros(26)
        score[8:14] = 2
        score[15:17] = 2
        score[19:22] = 2
        # Only three qualifying consecutive complete windows produce an event.
        self.assertEqual(transitions(score, 1, (10, 25), 3),
                         [{"time_s": 13, "evidence_start_s": 11},
                          {"time_s": 22, "evidence_start_s": 20}])
        score[20] = np.nan
        self.assertEqual(transitions(score, 1, (10, 25), 3),
                         [{"time_s": 13, "evidence_start_s": 11}])

    def test_sb112_offline_gap_is_not_negative_exposure_time(self):
        base = datetime(2022, 3, 10, tzinfo=timezone(timedelta(hours=2)))
        rows = [(base + timedelta(seconds=n), value) for n, value in
                ((0, 5), (20, 5), (40, 0), (400, 5), (420, 5))]
        scored = sb112_day(rows, 1, 2, 60, 300)
        self.assertAlmostEqual(scored["coverage_h"], 60 / 3600)
        self.assertEqual(scored["events"], 2)
        self.assertEqual(scored["segments"], 2)


if __name__ == "__main__":
    unittest.main()
