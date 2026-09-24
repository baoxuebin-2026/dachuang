"""Protocol tests for leakage-sensitive and boundary-sensitive behavior."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "run_m1_uci309", Path(__file__).resolve().parents[1] / "scripts" / "run_m1_uci309.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProtocolTests(unittest.TestCase):
    def test_pre_release_alarm_persisting_past_sixty_is_not_a_detection(self):
        scores = np.zeros(240)
        scores[50:100] = 2.0
        events = MODULE.evidence_events(scores, np.ones(240, dtype=bool), 1.0)
        outcome = MODULE.score_events(events)
        self.assertEqual(outcome["false_file"], 1)
        self.assertEqual(outcome["detected"], 0)
        self.assertEqual(outcome["already_g2_at_release"], 1)

    def test_missing_second_breaks_three_second_soft_evidence(self):
        scores = np.zeros(240)
        scores[58:64] = 2.0
        valid = np.ones(240, dtype=bool)
        valid[60] = False
        events = MODULE.evidence_events(scores, valid, 1.0)
        self.assertEqual(events["g2_events"], [64])
        self.assertEqual(events["invalid_seconds"], 1)
        self.assertEqual(MODULE.score_events(events)["delay_s"], 4)

    def test_future_values_cannot_change_prior_alerts(self):
        scores = np.zeros(240)
        scores[40:43] = 2.0
        valid = np.ones(240, dtype=bool)
        original = MODULE.evidence_events(scores, valid, 1.0)
        scores[150:240] = 1e6
        changed = MODULE.evidence_events(scores, valid, 1.0)
        self.assertEqual(original["g2_events"][0], 43)
        self.assertEqual(original["g2_events"][0], changed["g2_events"][0])

    def test_pre_release_seed_cannot_become_post_release_detection(self):
        scores = np.zeros(240)
        scores[56:100] = 2.0
        events = MODULE.evidence_events(scores, np.ones(240, dtype=bool), 1.0, duration_s=5)
        self.assertEqual(events["g2_events"][0], 61)
        outcome = MODULE.score_events(events)
        self.assertEqual(outcome["pre_release_seeded_events"], 1)
        self.assertEqual(outcome["detected"], 0)


if __name__ == "__main__":
    unittest.main()
