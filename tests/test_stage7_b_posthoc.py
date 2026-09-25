import unittest

from scripts.diagnose_stage7_b import ethylene_only_onsets, first_new_alarm, longest_above


class PosthocBoundaries(unittest.TestCase):
    def test_ethylene_onset_requires_eligible_prior_and_dwell(self):
        runs = [{"state": "both_zero", "start_s": 0, "duration_s": 100},
                {"state": "target_zero_ethylene_present", "start_s": 100, "duration_s": 100},
                {"state": "both_zero", "start_s": 200, "duration_s": 40},
                {"state": "target_zero_ethylene_present", "start_s": 240, "duration_s": 100}]
        self.assertEqual(ethylene_only_onsets(runs, (0, 300)), [100])


    def test_isolated_spikes_do_not_count_as_three_second_alarm(self):
        self.assertEqual(longest_above([1, 5, 2, 5, 2], 3), 1)
        self.assertIsNotNone(first_new_alarm([{"time_s": 107, "evidence_start_s": 105}], 100, 60))
        self.assertIsNone(first_new_alarm([{"time_s": 101, "evidence_start_s": 99}], 100, 60))


if __name__ == "__main__":
    unittest.main()
