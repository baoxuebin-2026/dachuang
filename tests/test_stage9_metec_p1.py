"""Protocol tests for the METEC P1 date-grouped diagnostic."""

import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "run_stage9_metec_p1",
    Path(__file__).resolve().parents[1] / "scripts" / "run_stage9_metec_p1.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetecP1Tests(unittest.TestCase):
    def test_held_out_date_does_not_set_its_own_threshold(self):
        per_date = {
            "2024-01-01": {"release": [9.0, 10.0], "outside_release": [100.0, 101.0]},
            "2024-01-02": {"release": [3.0, 4.0], "outside_release": [1.0, 2.0]},
        }
        rows = MODULE.leave_one_date_out(per_date, min_release=2, min_outside=2)
        first = next(row for row in rows if row["held_out_date"] == "2024-01-01")
        self.assertLess(first["p95_threshold_ppm"], 3.0)

    def test_day_eligibility_requires_both_states(self):
        per_date = {
            "2024-01-01": {"release": [2.0, 3.0], "outside_release": [1.0, 1.1]},
            "2024-01-02": {"release": [], "outside_release": [0.5, 0.6]},
        }
        rows = MODULE.leave_one_date_out(per_date, min_release=2, min_outside=2)
        self.assertEqual([row["held_out_date"] for row in rows], ["2024-01-01"])

    def test_threshold_uses_strict_greater_than(self):
        per_date = {
            "2024-01-01": {"release": [1.0, 2.0], "outside_release": [1.0, 2.0]},
            "2024-01-02": {"release": [1.0, 2.0], "outside_release": [1.0, 2.0]},
        }
        rows = MODULE.leave_one_date_out(per_date, min_release=2, min_outside=2)
        self.assertEqual(rows[0]["p99_release_fraction"], 0.5)

    def test_bootstrap_is_deterministic(self):
        first = MODULE.bootstrap_median_interval([0.1, 0.2, 0.9], 123, 100)
        second = MODULE.bootstrap_median_interval([0.1, 0.2, 0.9], 123, 100)
        self.assertEqual(first, second)

    def test_fixed_mst_mapping(self):
        local = datetime(2024, 3, 10, 3, 1, 29)
        mapped = MODULE.local_to_utc(local, "fixed_mst")
        self.assertEqual(mapped, datetime(2024, 3, 10, 10, 1, 29, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
