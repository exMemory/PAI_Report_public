"""Japanese video overlays and API event timing without network requests."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_pk_runner import EpisodeResult
import render_ai_pk_video as video


class TestVideoEvents(unittest.TestCase):
    def setUp(self):
        self.decision = {
            "sent_at": 0.1, "received_at": 1.1, "applied_at": 1.1,
            "reported_latency_s": 1.0, "latency_ms": 1000,
            "action": "GOTO_STANCE", "dist_to_stance_m": 0.15,
            "yaw_err_deg": 5.0, "t": 0.1,
        }

    def test_send_wait_receive_boundaries(self):
        decisions = [self.decision]
        self.assertEqual(video._api_state(0, decisions)[0], "idle")
        self.assertEqual(video._api_state(.1, decisions)[0], "waiting")
        self.assertEqual(video._api_state(1.09, decisions)[0], "waiting")
        self.assertEqual(video._api_state(1.1, decisions)[0], "received")
        self.assertEqual(video._api_state(1.5, decisions)[2], 1)

    def test_japanese_send_receive_graphics_have_distinct_colors(self):
        snapshot = {"sim_time": .15, "action": "HOLD", "goal": False, "fallen": False}
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        sent = video._hud(blank, snapshot, "Gemini（実API）", [self.decision], True)
        snapshot["sim_time"] = 1.15
        received = video._hud(blank, snapshot, "Gemini（実API）", [self.decision], True)
        np.testing.assert_array_equal(sent[72, 10], [70, 190, 255])
        np.testing.assert_array_equal(received[72, 10], [95, 235, 165])
        self.assertGreater(np.std(sent), 10)

    def test_late_response_is_not_shown_as_received_before_end(self):
        self.decision["received_at"] = 35.0
        self.decision["applied_at"] = None
        self.assertEqual(video._api_state(30, [self.decision])[0], "waiting")
        card = video._decision_card(np.zeros((480, 640, 3), dtype=np.uint8),
                                    1, 1, self.decision)
        self.assertEqual(card.shape, (480, 640, 3))

    def test_five_final_screenshots_have_correct_success_count(self):
        finals = [np.full((480, 640, 3), i * 40, dtype=np.uint8) for i in range(5)]
        results = [{"goal": i < 3, "fallen": i == 2, "kick_times": [1]}
                   for i in range(5)]
        grid = video._result_grid(finals, results, "Gemini ER2")
        self.assertEqual(grid.shape, (480, 640, 3))
        for i in range(5):
            row, col = divmod(i, 3)
            np.testing.assert_array_equal(grid[62 + row * 204 + 60, 8 + col * 210 + 50],
                                          [i * 40] * 3)

    def test_five_trial_rules_compilation_and_final_count(self):
        decider = SimpleNamespace(api_calls=0)
        calls = []

        def episode(decider, **kwargs):
            calls.append(kwargs)
            decider.api_calls += 1
            frame = np.full((480, 640, 3), 70, dtype=np.uint8)
            obs = SimpleNamespace(sim_time=.1, goal_scored=len(calls) < 3,
                                  robot_fallen=False)
            kwargs["frame_callback"](
                SimpleNamespace(render_camera=lambda camera: frame), obs, "HOLD")
            return EpisodeResult(goal=obs.goal_scored, end_time=.1, end_reason="goal"
                                 if obs.goal_scored else "timeout")

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(video, "run_ai_episode", side_effect=episode):
            result = video.record_five_trials(
                decider, Path(directory), "test_", "Gemini ER2", "test-model")
            self.assertEqual(result["trials_count"], 5)
            self.assertEqual(result["successes"], 2)
            self.assertEqual(result["api_calls"], 5)
            self.assertEqual(result["video"]["frame_count"], 16 * 30 + 5)
            self.assertTrue(all(c["max_time"] == 60 and c["stop_on_fall"] for c in calls))
            self.assertTrue(Path(result["final_grid"]).is_file())

    def test_composition_is_exactly_sixty_seconds_with_late_response(self):
        self.decision["applied_at"] = None
        result = EpisodeResult(decisions=[self.decision], end_reason="timeout",
                               end_time=.2, final_ball=(.25, -.09, .1))
        frames = [np.full((480, 640, 3), 50, dtype=np.uint8) for _ in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = video._compose_video(root, frames, result, provider_note="Gemini（実API）", live=True)
            stats = video._verify_and_contact_sheet(path, root)
            self.assertEqual(stats["frame_count"], 1800)
            self.assertEqual(stats["duration_sec"], 60)
            with imageio.get_reader(str(root / "episode.mp4")) as reader:
                self.assertEqual(reader.count_frames(), 6)


if __name__ == "__main__":
    unittest.main()
