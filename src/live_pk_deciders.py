"""Live API decision makers for the RL-backed closed-loop action menu."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from ai_pk_runner import ACTIONS


INSTRUCTIONS = (
    "Control a simulated Unitree G1 penalty kick using only the supplied state. "
    "SUCCESS means the ball actually crosses into the goal BEFORE time_limit_s "
    "(60 seconds in these trials), not merely starting a kick before the deadline. "
    "No goal by the deadline is failure. You may kick repeatedly. A fall ends "
    "stop-on-fall trials immediately. Do not trade an obvious fall for urgency. "
    "remaining_time_s includes this API response wait, movement, kick execution "
    "and the ball's travel to goal. The clock never resets between shots. "
    "Choose one action: GOTO_STANCE continuously walks toward the LIVE ball's "
    "stance point AND corrects heading; you need not issue ALIGN as a mandatory "
    "separate step. ALIGN stands in place and corrects heading ONLY: it cannot "
    "reduce position error or guarantee both feet will stay planted. "
    "KICK immediately starts the shared strike; HOLD stands and holds heading. "
    "The RL low layer maintains balance, but does not guarantee a successful kick. "
    "Calibrated reference tolerances: approach distance <=0.07m, heading error "
    "<2.3deg, yaw rate <11.5deg/s and both feet planted are favorable for KICK. "
    "These are guidelines, not requirements to achieve a perfect stationary pose. "
    "Do not automatically ALIGN just because one transient measurement is imperfect. "
    "Use ALIGN only when close enough and a heading or rotation correction is "
    "actually needed. If already close and adequately oriented, favor a timely "
    "KICK over repeated ALIGN. If too far away, use GOTO_STANCE, not ALIGN. "
    "previous_decision reports the last applied choice and its observation. "
    "Compare distance/heading to that observation before repeating ALIGN; "
    "do not keep aligning without progress. current_action is what will execute "
    "while this request is pending. last_api_latency_s is the last measured "
    "request duration, NOT a guarantee of the next request duration. "
    "Budget that response delay plus movement and ball travel against remaining_time_s. "
    "As the deadline approaches, avoid optional extra alignment when a viable "
    "shot is available, but never kick solely because time is low if the ball "
    "is out of reach or the robot is visibly unstable. "
    "HOLD after kick_taken, robot_fallen or goal_scored. "
    "kick_taken refers to the active shot, not the entire episode. After an "
    "unsuccessful shot stops and recovery finishes, it resets to false and "
    "you must approach the ball's new position and attempt another shot. "
    "These are state-based guidelines, not a timed action sequence. "
    "During API inference physics continues under current_action, so your "
    "observation will be stale when applied. Return concise JSON, no discussion; "
    "reasoning_note must be at most 12 words."
)


class LiveDecisionError(RuntimeError):
    """A live provider request or its decision failed; no fallback is applied."""


def _parse(text: Any) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise LiveDecisionError("Live provider returned no decision text.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[-1].strip() != "```":
            raise LiveDecisionError("Live provider returned an incomplete JSON fence.")
        cleaned = "\n".join(lines[1:-1])
    try:
        payload = json.loads(cleaned)
    except (ValueError, TypeError):
        raise LiveDecisionError("Live provider returned invalid decision JSON.") from None
    if not isinstance(payload, dict):
        raise LiveDecisionError("Live provider decision must be a JSON object.")
    return payload


def _action(payload: dict) -> str:
    action = payload.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        raise LiveDecisionError("Live provider returned an unknown action.")
    return action


def _reason(payload: dict) -> str:
    note = payload.get("reasoning_note", "")
    if not isinstance(note, str):
        raise LiveDecisionError("Live provider reasoning_note must be a string.")
    return note


class GeminiLiveDecider:
    def __init__(self, client: Any, model_name: str = "gemini-robotics-er-2-preview"):
        self.client = client
        self.model_name = model_name
        self.api_calls = 0
        self.last_latency_s = 0.0
        self.last_reasoning = ""

    def decide(self, state: dict) -> str:
        self.api_calls += 1
        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=INSTRUCTIONS + "\nReturn JSON with action and reasoning_note.\n"
                + json.dumps(state),
                config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
        except Exception as exc:
            status = getattr(exc, "code", None)
            safe_status = f" (status {status})" if isinstance(status, int) else ""
            raise LiveDecisionError(f"Gemini API request failed{safe_status}; no fallback used.") from None
        finally:
            self.last_latency_s = time.perf_counter() - start
        payload = _parse(getattr(response, "text", None))
        action = _action(payload)
        self.last_reasoning = _reason(payload)
        return action


class JEVLiveDecider:
    def __init__(self, client: Any, model_name: str = "typesafe/jev-router"):
        self.client = client
        self.model_name = model_name
        self.api_calls = 0
        self.last_latency_s = 0.0
        self.last_reasoning = ""

    def decide(self, state: dict) -> str:
        self.api_calls += 1
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": INSTRUCTIONS + (
                        " Return JSON with a candidates array containing exactly one entry "
                        "per allowed action. Each entry must have action, score (finite "
                        "number between 0 and 1), and reasoning_note (at most 12 words). "
                        "Highest score wins. Output only the complete JSON object, no prose."
                    )},
                    {"role": "user", "content": json.dumps(state)},
                ],
                temperature=0,
                max_tokens=1600,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            safe_status = f" (status {status})" if isinstance(status, int) else ""
            raise LiveDecisionError(f"JEV API request failed{safe_status}; no fallback used.") from None
        finally:
            self.last_latency_s = time.perf_counter() - start
        choices = getattr(response, "choices", None)
        if not choices:
            raise LiveDecisionError("JEV API response has no choices.")
        if getattr(choices[0], "finish_reason", None) == "length":
            raise LiveDecisionError("JEV API response reached its output limit; no fallback used.")
        payload = _parse(getattr(choices[0].message, "content", None))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(ACTIONS):
            raise LiveDecisionError("JEV must score exactly one candidate per action.")
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise LiveDecisionError("JEV candidate must be a JSON object.")
            action = _action(candidate)
            if action in seen:
                raise LiveDecisionError("JEV returned duplicate action candidates.")
            seen.add(action)
            score = candidate.get("score")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0.0 <= score <= 1.0):
                raise LiveDecisionError("JEV candidate score must be finite and between 0 and 1.")
            _reason(candidate)
        best = max(candidates, key=lambda candidate: candidate["score"])
        self.last_reasoning = _reason(best)
        return _action(best)
