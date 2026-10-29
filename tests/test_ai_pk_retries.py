"""Repeat-shot lifecycle and recovery safety checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_ai_pk_latency import FakeEnv, FakeExecutor, runner


class RetryEnv(FakeEnv):
    goal_after_second = False
    moving = False

    def __init__(self):
        super().__init__()
        self.shots = 0

    def get_observation(self):
        obs = super().get_observation()
        obs.ball_vel = (0.2 if self.moving else 0.0, 0.0, 0.0)
        obs.goal_scored = self.goal_after_second and self.shots >= 2
        return obs


class RetryExecutor(FakeExecutor):
    def set_action(self, action):
        super().set_action(action)
        if action == "KICK":
            self.env.shots += 1

    def rearm_for_retry(self):
        self.kick_taken = False
        self.action = "GOTO_STANCE"
        return True


class TestRetries(unittest.TestCase):
    def run_episode(self, *, goal=False, moving=False, fall_at=float("inf"), max_time=2.0,
                    stop_on_fall=False):
        FakeExecutor.applied = []
        RetryEnv.goal_after_second = goal
        RetryEnv.moving = moving
        RetryEnv.fall_at = fall_at
        states = []

        def decide(state):
            states.append(state.copy())
            return "KICK" if state["dist_to_stance_m"] < 0.07 else "GOTO_STANCE"

        def build_state(env, params, kick_taken):
            return {
                "kick_taken": kick_taken,
                "dist_to_stance_m": 0.2 if env.shots == 1 and env.t < 0.8 else 0.01,
                "yaw_err_deg": 0.0, "ball_pos": env.get_observation().ball_pos,
            }

        with patch.object(runner, "SoccerPKEnvRL", RetryEnv), \
                patch.object(runner, "RLActionExecutor", RetryExecutor), \
                patch.object(runner, "build_state", side_effect=build_state):
            result = runner.run_ai_episode(SimpleNamespace(decide=decide), max_time=max_time,
                                           stop_on_fall=stop_on_fall)
        return result, states

    def test_stopped_shot_reapproaches_then_scores(self):
        result, states = self.run_episode(goal=True)
        self.assertEqual(FakeExecutor.applied[0], "KICK")
        self.assertEqual(FakeExecutor.applied[-1], "KICK")
        self.assertTrue(all(action == "GOTO_STANCE"
                            for action in FakeExecutor.applied[1:-1]))
        self.assertGreater(len(FakeExecutor.applied), 2)
        self.assertEqual(len(result.kick_times), 2)
        self.assertEqual(result.kick_time, result.kick_times[0])
        self.assertTrue(result.goal)
        self.assertEqual(result.end_reason, "goal")
        self.assertTrue(all(not state["kick_taken"] for state in states))
        self.assertEqual(result.decisions[-1]["shot_number"], 2)
        self.assertIsNone(states[0]["previous_decision"])
        self.assertEqual(states[1]["previous_decision"]["action"], "KICK")
        self.assertEqual(states[1]["current_action"], "GOTO_STANCE")
        self.assertEqual(states[1]["kick_count"], 1)
        self.assertLess(states[1]["remaining_time_s"], states[0]["remaining_time_s"])
        self.assertEqual(result.decisions[0]["input_state"]["remaining_time_s"],
                         states[0]["remaining_time_s"])
        self.assertEqual(result.decisions[-1]["input_state"], states[-1])

    def test_retries_share_overall_timeout(self):
        result, _ = self.run_episode()
        self.assertGreaterEqual(len(result.kick_times), 2)
        self.assertEqual(result.end_reason, "timeout")
        self.assertLessEqual(result.final_ball[0], 2.002)

    def test_moving_ball_does_not_trigger_retry_after_eight_seconds(self):
        result, _ = self.run_episode(moving=True, max_time=9.0)
        self.assertEqual(len(result.kick_times), 1)
        self.assertEqual(result.end_reason, "timeout")

    def test_fallen_robot_does_not_retry(self):
        result, _ = self.run_episode(fall_at=0.2)
        self.assertEqual(len(result.kick_times), 1)
        self.assertEqual(result.end_reason, "fall_after_kick")

    def test_stop_on_fall_ends_immediately_even_with_moving_ball(self):
        result, _ = self.run_episode(fall_at=.2, moving=True, stop_on_fall=True)
        self.assertEqual(result.end_reason, "fall_after_kick")
        self.assertAlmostEqual(result.end_time, .2)
        self.assertEqual(len(result.kick_times), 1)

    def test_executor_rearm_waits_for_full_recovery(self):
        env = SimpleNamespace(data=SimpleNamespace(time=1.0))
        ex = runner.RLActionExecutor(env)
        self.assertFalse(ex.rearm_for_retry())
        ex.kick_taken = True
        ex.kick_time = 0.2
        ex.kick = object()
        ex._recover_start = 0.5
        self.assertFalse(ex.rearm_for_retry())
        ex.kick = None
        self.assertFalse(ex.rearm_for_retry())
        env.data.time = 0.5 + ex.p.recover_time + 0.002
        self.assertTrue(ex.rearm_for_retry())
        self.assertFalse(ex.kick_taken)
        self.assertEqual(ex.action, "GOTO_STANCE")
        self.assertIsNone(ex._recover_start)
        self.assertFalse(ex.rearm_for_retry())


if __name__ == "__main__":
    unittest.main()
