"""Event matching and split guards for the development comparison."""

import unittest

import numpy as np

from scripts.run_stage9_b2_a_dev_comparison import feature_matrix, score_alerts


class AlertScoringTests(unittest.TestCase):
    def test_exact_horizon_is_included_but_alarm_at_onset_is_not(self):
        t = np.array([600, 660, 960])
        e = np.array([960])
        y = np.array([1, 1, 0])
        out = score_alerts(t, t, y, e, e, np.array([1., 0., 1.]), .5)
        self.assertEqual(out["matched_events"], 1)
        self.assertEqual(out["first_lead_seconds"], [360])
        self.assertEqual(out["false_new_alerts_background"], 1)

    def test_alarm_after_event_cannot_be_early_detection(self):
        t = np.array([600, 660, 720, 780, 840, 900, 960, 1020])
        y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        score = np.array([0, 0, 0, 1, 1, 0, 1, 0], dtype=float)
        out = score_alerts(t, t, y, np.array([900]), np.array([900]), score, .5)
        self.assertEqual(out["matched_events"], 1)
        self.assertEqual(out["first_lead_seconds"], [120])
        self.assertEqual(out["false_new_alerts_background"], 1)  # 960 is after event.

    def test_continuous_alarm_before_six_minute_window_does_not_count(self):
        t = np.arange(600, 1381, 60)
        e = 1260
        y = ((t >= e - 360) & (t < e)).astype(int)
        score = np.ones(len(t))
        out = score_alerts(t, t, y, np.array([e]), np.array([e]), score, .5)
        self.assertEqual(out["evaluable_events"], 1)
        self.assertEqual(out["matched_events"], 0)
        self.assertEqual(out["false_new_alerts_background"], 1)

    def test_gap_resets_new_alert_and_no_event_twice(self):
        t = np.array([600, 660, 780, 840, 900, 960])
        e = np.array([1000])
        y = (t >= e[0] - 360).astype(int)
        score = np.array([0, 1, 1, 1, 0, 1], dtype=float)
        out = score_alerts(t, t, y, e, e, score, .5)
        self.assertEqual(out["new_alert_episodes"], 3)
        self.assertEqual(out["matched_events"], 1)
        self.assertEqual(out["duplicate_or_unmatched_positive_new_alerts"], 2)

    def test_features_never_look_forward(self):
        x = np.linspace(.1, .9, 1000).astype(np.float32)
        before = feature_matrix(x, np.array([659]))
        x[660:] = 99
        after = feature_matrix(x, np.array([659]))
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
