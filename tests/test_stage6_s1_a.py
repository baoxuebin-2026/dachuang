"""Check scenario geometry, timing and safety-preserving unknown handling."""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.run_stage6_s1_a import (MATRIX, fuse, gas_states, locate, position_at,
                                      signed_distance, simulate)


class S1ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(Path("configs/stage6_s1_fixed_a.json").read_text())

    def test_geometry_and_ambiguous_boundaries(self):
        zone = self.cfg["control_zone"]
        self.assertEqual(signed_distance(-1.0, 1, zone), 1.0)
        self.assertEqual(signed_distance(0.5, 1, zone), -0.5)
        self.assertAlmostEqual(signed_distance(-1, -1, zone), np.sqrt(2))
        self.assertEqual(locate(0, 1, 0), (2, "valid"))
        self.assertEqual(locate(1, 1, 0), (1, "valid"))
        self.assertEqual(locate(1, 1, 0.2)[1], "boundary_uncertain")
        self.assertEqual(locate(-0.5, 1, 0.2), (2, "valid"))

    def test_monotone_matrix_and_independent_sources_survive_unknown(self):
        for g in range(3):
            for p in range(3):
                self.assertEqual(fuse(g, p, same_zone=True)["joint_level"], MATRIX[g][p])
                if g < 2:
                    self.assertLessEqual(MATRIX[g][p], MATRIX[g+1][p])
                if p < 2:
                    self.assertLessEqual(MATRIX[g][p], MATRIX[g][p+1])
        self.assertEqual(fuse(None, 2, same_zone=True, gas_quality="simulated_dropout")
                         ["known_lower_bound"], 2)
        self.assertEqual(fuse(2, None, same_zone=True, person_quality="simulated_dropout")
                         ["known_lower_bound"], 2)
        self.assertIsNone(fuse(None, None, same_zone=True, gas_quality="bad",
                               person_quality="bad")["known_lower_bound"])
        mismatch = fuse(2, 2, same_zone=False)
        self.assertIsNone(mismatch["joint_level"])
        self.assertEqual((mismatch["gas_level"], mismatch["person_level"]), (2, 2))

    def test_missing_second_resets_continuous_soft_evidence(self):
        scores = np.zeros(240)
        valid = np.ones(240, dtype=bool)
        scores[28:34] = 10
        states = gas_states(scores, valid, threshold=1, duration_s=3, missing_at={31})
        self.assertEqual([states[t][0] for t in range(29, 35)], [1, 1, None, 1, 1, 2])
        one_spike = np.zeros(240)
        one_spike[29] = 10
        self.assertNotIn(2, [value for value, _ in gas_states(one_spike, valid, 1, 3).values()])

    def test_fixed_path_and_wrong_zone_control(self):
        frames = self.cfg["keyframes_xy"]["possible_overlap"]
        self.assertEqual(position_at(frames, 98), (0.5, 1.0))
        scores = np.full(240, 10.0)
        valid = np.ones(240, dtype=bool)
        selected = {"threshold": 1.0}
        normal, trace = simulate(scores, valid, selected, "S2_possible_overlap", self.cfg,
                                 buffer_m=1, error_bound_m=0, duration_s=3,
                                 include_timeline=True)
        mismatch, wrong = simulate(scores, valid, selected, "S4_wrong_zone", self.cfg,
                                   buffer_m=1, error_bound_m=0, duration_s=3,
                                   include_timeline=True)
        self.assertGreater(normal["joint_l3_seconds"], mismatch["joint_l3_seconds"])
        self.assertTrue(all(row["joint_level"] is None and row["gas_level"] == 2
                            and row["person_level"] == 2
                            for row in wrong if 100 <= row["completed_time_s"] < 110))
        self.assertEqual(mismatch["zone_mismatch_seconds"], 10)
        # Later changes to the gas trace cannot change already emitted fusion rows.
        altered = scores.copy()
        altered[130:] = 0.0
        _, trace_alt = simulate(altered, valid, selected, "S2_possible_overlap", self.cfg,
                                buffer_m=1, error_bound_m=0, duration_s=3,
                                include_timeline=True)
        self.assertEqual([x["joint_level"] for x in trace if x["completed_time_s"] <= 130],
                         [x["joint_level"] for x in trace_alt if x["completed_time_s"] <= 130])


if __name__ == "__main__":
    unittest.main()
