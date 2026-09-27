"""Small independent alarm-trace boundary checks for D1."""

import unittest

import numpy as np

from scripts.diagnose_stage9_b2_d1_development import trace_alarms


class TraceTests(unittest.TestCase):
    def test_early_lead_counts_and_gap_only_control(self):
        times = np.array([600, 660, 780, 840, 900, 960])
        events = np.array([1000])
        labels = ((times >= 640) & (times < 1000)).astype(int)
        trace = trace_alarms(times, labels, events, np.ones(len(times)), .5, [60, 180])
        self.assertEqual(trace["new_alert_reason"], {
            "first_partition_anchor": 1, "nonadjacent_eligible_anchor": 1,
        })
        self.assertEqual(trace["matched_events_any_lead"], 1)
        self.assertEqual(trace["matched_events_at_least_lead_seconds"], {"60": 1, "180": 1})
        self.assertEqual(trace["false_new_alerts_background"], 1)

    def test_five_seconds_is_not_a_minute_of_early_warning(self):
        times = np.array([635, 995])
        events = np.array([1000])
        labels = np.array([0, 1])
        trace = trace_alarms(times, labels, events, np.array([0., 1.]), .5, [60, 180])
        self.assertEqual(trace["matched_events_any_lead"], 1)
        self.assertEqual(trace["event_records"][0]["lead_seconds"], 5)
        self.assertEqual(trace["matched_events_at_least_lead_seconds"], {"60": 0, "180": 0})


if __name__ == "__main__":
    unittest.main()
