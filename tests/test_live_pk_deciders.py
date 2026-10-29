"""Offline contract tests; these do not substitute for live API trials."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from live_pk_deciders import ACTIONS, GeminiLiveDecider, JEVLiveDecider, LiveDecisionError
from live_pk_deciders import INSTRUCTIONS


class TestLiveDeciders(unittest.TestCase):
    def test_shared_prompt_explains_deadline_action_limits_and_progress(self):
        for required in (
            "ball actually crosses into the goal BEFORE time_limit_s",
            "ball's travel to goal", "clock never resets",
            "ALIGN stands in place", "cannot reduce position error",
            "not keep aligning without progress", "last_api_latency_s",
            "never kick solely because time is low",
        ):
            self.assertIn(required, INSTRUCTIONS)

    def test_both_providers_receive_the_same_deadline_and_history(self):
        state = {
            "remaining_time_s": 8.0, "time_limit_s": 60,
            "current_action": "ALIGN", "last_api_latency_s": 4.0,
            "previous_decision": {"action": "ALIGN", "dist_to_stance_m": .04},
        }
        gemini_client = SimpleNamespace(models=Mock())
        gemini_client.models.generate_content.return_value = SimpleNamespace(
            text='{"action":"KICK","reasoning_note":"ready"}')
        GeminiLiveDecider(gemini_client).decide(state)
        gemini_prompt = gemini_client.models.generate_content.call_args.kwargs["contents"]
        self.assertTrue(gemini_prompt.startswith(INSTRUCTIONS))
        self.assertTrue(gemini_prompt.endswith(json.dumps(state)))
        jev = self.jev(self.candidates())
        jev.decide(state)
        messages = jev.client.chat.completions.create.call_args.kwargs["messages"]
        self.assertTrue(messages[0]["content"].startswith(INSTRUCTIONS))
        self.assertEqual(json.loads(messages[1]["content"]), state)

    def test_gemini_calls_real_client_interface(self):
        client = SimpleNamespace(models=Mock())
        client.models.generate_content.return_value = SimpleNamespace(
            text='{"action":"ALIGN","reasoning_note":"settle first"}')
        decider = GeminiLiveDecider(client)
        self.assertEqual(decider.decide({"sim_time": 0.0}), "ALIGN")
        self.assertEqual(decider.api_calls, 1)
        self.assertGreaterEqual(decider.last_latency_s, 0.0)
        self.assertEqual(decider.last_reasoning, "settle first")
        self.assertEqual(client.models.generate_content.call_args.kwargs["model"],
                         "gemini-robotics-er-2-preview")

    def jev(self, candidates):
        completions = Mock()
        completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps({"candidates": candidates})))])
        return JEVLiveDecider(SimpleNamespace(
            chat=SimpleNamespace(completions=completions)))

    def candidates(self):
        return [{"action": action, "score": index / 4, "reasoning_note": action}
                for index, action in enumerate(ACTIONS)]

    def test_jev_uses_highest_provider_score(self):
        decider = self.jev(self.candidates())
        self.assertEqual(decider.decide({}), "HOLD")
        self.assertEqual(decider.api_calls, 1)
        self.assertEqual(decider.last_reasoning, "HOLD")

    def test_jev_truncated_response_has_explicit_error(self):
        decider = self.jev(self.candidates())
        decider.client.chat.completions.create.return_value.choices[0].finish_reason = "length"
        with self.assertRaisesRegex(LiveDecisionError, "output limit"):
            decider.decide({})

    def test_invalid_scores_or_coverage_are_rejected(self):
        for score in (None, True, float("nan"), float("inf"), -1, 2, "0.5"):
            with self.subTest(score=score):
                candidates = self.candidates()
                candidates[0]["score"] = score
                with self.assertRaises(LiveDecisionError):
                    self.jev(candidates).decide({})
        for mode in ("missing", "duplicate", "unknown"):
            candidates = self.candidates()
            if mode == "missing":
                candidates.pop()
            else:
                candidates[0]["action"] = "HOLD" if mode == "duplicate" else "FIRE"
            with self.subTest(mode=mode), self.assertRaises(LiveDecisionError):
                self.jev(candidates).decide({})

    def test_invalid_gemini_decisions_are_rejected(self):
        for text in ("[]", "bad", '{"action":"FIRE"}', '{"action":true}',
                     '{"action":"KICK","reasoning_note":2}'):
            client = SimpleNamespace(models=Mock())
            client.models.generate_content.return_value = SimpleNamespace(text=text)
            with self.subTest(text=text), self.assertRaises(LiveDecisionError):
                GeminiLiveDecider(client).decide({})

    def test_errors_do_not_expose_provider_exception_or_fallback(self):
        for provider in ("gemini", "jev"):
            method = Mock(side_effect=RuntimeError("sensitive-provider-detail"))
            if provider == "gemini":
                decider = GeminiLiveDecider(SimpleNamespace(
                    models=SimpleNamespace(generate_content=method)))
            else:
                decider = JEVLiveDecider(SimpleNamespace(
                    chat=SimpleNamespace(completions=SimpleNamespace(create=method))))
            with self.subTest(provider=provider), self.assertRaises(LiveDecisionError) as error:
                decider.decide({})
            self.assertNotIn("sensitive-provider-detail", str(error.exception))
            self.assertEqual(decider.api_calls, 1)
            self.assertGreaterEqual(decider.last_latency_s, 0.0)


if __name__ == "__main__":
    unittest.main()
