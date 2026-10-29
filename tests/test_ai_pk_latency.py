"""Focused checks for physics and camera updates during live inference."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ai_pk_runner as runner


class FakeEnv:
    fall_at = float("inf")

    def __init__(self):
        self.t = 0.0

    def reset(self, **kwargs):
        self.t = 0.0

    def get_observation(self):
        return SimpleNamespace(
            sim_time=self.t, robot_fallen=self.t >= self.fall_at,
            goal_scored=False, ball_vel=(1.0, 0.0, 0.0),
            ball_pos=(self.t, 0.0, 0.1),
        )

    def close(self):
        pass


class FakeExecutor:
    applied = []

    def __init__(self, env, params):
        self.env = env
        self.action = "HOLD"
        self.kick_taken = False
        self.kick_time = None

    def step(self):
        self.env.t += 0.002

    def set_action(self, action):
        self.applied.append(action)
        self.action = action
        if action == "KICK":
            self.kick_taken = True
            self.kick_time = self.env.t


class TestLatency(unittest.TestCase):
    def run_episode(self, latency, fall_at=float("inf")):
        frames = []
        FakeEnv.fall_at = fall_at
        FakeExecutor.applied = []
        decider = SimpleNamespace(last_latency_s=latency, decide=lambda state: "KICK")
        state = {"dist_to_stance_m": 0.01, "yaw_err_deg": 0.0, "ball_pos": (0, 0, 0)}
        with patch.object(runner, "SoccerPKEnvRL", FakeEnv), \
                patch.object(runner, "RLActionExecutor", FakeExecutor), \
                patch.object(runner, "build_state", return_value=state):
            result = runner.run_ai_episode(
                decider, max_time=0.1,
                frame_callback=lambda env, obs, action: frames.append((obs.sim_time, action)),
            )
        return result, frames

    def test_fall_during_inference_never_applies_kick(self):
        result, frames = self.run_episode(0.08, fall_at=0.04)
        self.assertEqual(FakeExecutor.applied, [])
        self.assertTrue(result.fallen)
        self.assertAlmostEqual(result.fall_time, 0.04)
        self.assertIsNone(result.kick_time)
        self.assertIsNone(result.decisions[0]["applied_at"])
        self.assertEqual(result.end_reason, "fall_before_kick")
        self.assertGreaterEqual(len(frames), 2)
        self.assertTrue(all(action == "HOLD" for _, action in frames))
        self.assertGreater(result.max_ball_x, 0)

    def test_timeout_during_inference_never_applies_kick(self):
        result, frames = self.run_episode(1.0)
        self.assertEqual(FakeExecutor.applied, [])
        self.assertIsNone(result.decisions[0]["applied_at"])
        self.assertEqual(result.end_reason, "timeout")
        self.assertGreaterEqual(len(frames), 3)
        self.assertLessEqual(result.final_ball[0], 0.102)

    def test_frames_include_wait_before_action_is_applied(self):
        result, frames = self.run_episode(0.04)
        self.assertEqual(FakeExecutor.applied, ["KICK"])
        self.assertAlmostEqual(result.kick_time, 0.04)
        self.assertAlmostEqual(result.decisions[0]["applied_at"], 0.04)
        self.assertAlmostEqual(result.decisions[0]["sent_at"], 0.0)
        self.assertAlmostEqual(result.decisions[0]["received_at"], 0.04)
        self.assertIn("+00:00", result.decisions[0]["sent_wall_utc"])
        self.assertIn("+00:00", result.decisions[0]["received_wall_utc"])
        self.assertTrue(any(action == "HOLD" for _, action in frames))
        self.assertTrue(any(action == "KICK" for _, action in frames))


if __name__ == "__main__":
    unittest.main()
