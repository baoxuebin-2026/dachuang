"""Boundary tests for the METEC source-audit helpers."""

import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "audit_stage9_metec_source",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_stage9_metec_source.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetecAuditTests(unittest.TestCase):
    def test_fixed_mst_does_not_apply_daylight_saving_shift(self):
        value = "03/10/2024 03:01:29.021"
        parsed = MODULE.parse_observation_time(value, "fixed_mst")
        self.assertEqual(parsed, datetime(2024, 3, 10, 10, 1, 29, 21000, tzinfo=timezone.utc))

    def test_time_basis_choices_differ_after_spring_transition(self):
        value = "03/10/2024 03:01:29.021"
        fixed = MODULE.parse_observation_time(value, "fixed_mst")
        civil = MODULE.parse_observation_time(value, "america_denver")
        self.assertEqual((fixed - civil).total_seconds(), 3600)

    def test_touching_release_intervals_merge(self):
        first = datetime(2024, 2, 8, 0, 0, tzinfo=timezone.utc)
        second = datetime(2024, 2, 8, 0, 10, tzinfo=timezone.utc)
        third = datetime(2024, 2, 8, 0, 20, tzinfo=timezone.utc)
        self.assertEqual(MODULE.merge_intervals([(first, second), (second, third)]), [(first, third)])

    def test_rank_auc_handles_ties(self):
        self.assertEqual(MODULE.rank_auc([2.0, 2.0], [1.0, 2.0]), 0.75)


if __name__ == "__main__":
    unittest.main()
