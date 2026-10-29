"""Closed-loop decision interface on top of the RL low layer.

Background (see FIX_NOTES.md section 6): with the original design the AI
decided ONCE per episode and its decision was held -- an open loop on a
base that fell in ~1 s.  Now that the RL low layer keeps the robot upright,
the decision layer can work the way the assignment intends: observe the
state, choose the NEXT action, execute, observe again.  A fall ends the
episode, so every decision is a fresh chance to reason about what to do
next -- and slow (Gemini) vs fast (Jev) decision cadence becomes a
measurable property instead of a death sentence.

This module provides:
  * A discrete action menu for decision makers:
      GOTO_STANCE  walk to the stance point behind the ball (tracks the
                   live ball position)
      ALIGN        stand and correct heading toward the shot direction
      KICK         execute the scripted strike NOW (no automatic release
                   conditions -- kicking from a bad pose whiffs or falls,
                   and that outcome belongs to the decision maker)
      HOLD         stand still (heading held)
  * RLActionExecutor: executes the current action on SoccerPKEnvRL at
    500 Hz (RL policy for locomotion/balance, scripted two-leg strike
    with CoM shift for KICK, catch-stepping afterwards).
  * run_ai_episode: the episode runner with the new termination rules --
    goal / fall / timeout, except that after a kick is taken the ball is
    played out even if the robot falls. An upright robot can reapproach
    a stopped ball and kick again. Every decision is logged (time, state digest, action,
    latency) for the report tables.
  * RuleBasedDecider: a reference decision maker that uses ONLY the
    state dict the AI deciders receive, proving the interface suffices.

The validated controller in rl_pk_controller.py is left untouched; the
kick trajectory is re-encapsulated here as ScriptedKick so the AI can
trigger it on demand.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from soccer_pk_env import GOAL_LINE_X, PKStateObservation
from rl_pk_controller import (
    SoccerPKEnvRL,
    RLPKParams,
    _yaw_from_quat,
    _wrap_angle,
    _smoothstep,
)

ACTIONS = ("GOTO_STANCE", "ALIGN", "KICK", "HOLD")


def _shot_geometry(env: SoccerPKEnvRL, params: RLPKParams):
    """Stance point, distance to it, and heading error vs the shot line."""
    obs = env.get_observation()
    d = env.data
    ball = np.array(obs.ball_pos[:2])
    pelvis = np.array(obs.pelvis_pos[:2])
    yaw = _yaw_from_quat(*d.qpos[3:7])
    shot_yaw = math.atan2(0.0 - ball[1], GOAL_LINE_X - ball[0]) \
        + math.radians(params.aim_bias_deg)
    shot_dir = np.array([math.cos(shot_yaw), math.sin(shot_yaw)])
    perp = np.array([-shot_dir[1], shot_dir[0]])
    stance = ball - shot_dir * params.stance_dist + perp * params.stance_lateral
    to_stance = stance - pelvis
    return {
        "obs": obs, "ball": ball, "pelvis": pelvis, "yaw": yaw,
        "shot_yaw": shot_yaw, "stance": stance, "to_stance": to_stance,
        "dist_to_stance": float(np.linalg.norm(to_stance)),
        "yaw_err": _wrap_angle(shot_yaw - yaw),
        "yaw_rate": float(d.qvel[5]),
        "both_planted": bool(float(d.xpos[env.left_ankle_body][2]) < 0.055
                             and float(d.xpos[env.right_ankle_body][2]) < 0.055),
    }


def build_state(env: SoccerPKEnvRL, params: RLPKParams, kick_taken: bool) -> dict:
    """The state dict handed to decision makers (AI or rule-based)."""
    g = _shot_geometry(env, params)
    obs = g["obs"]
    return {
        "sim_time": obs.sim_time,
        "ball_pos": tuple(round(v, 3) for v in obs.ball_pos),
        "ball_vel": tuple(round(v, 3) for v in obs.ball_vel),
        "pelvis_pos": tuple(round(v, 3) for v in obs.pelvis_pos),
        "dist_to_stance_m": round(g["dist_to_stance"], 3),
        "yaw_err_deg": round(math.degrees(g["yaw_err"]), 1),
        "yaw_rate_deg_s": round(math.degrees(g["yaw_rate"]), 1),
        "both_feet_planted": g["both_planted"],
        "torso_roll_deg": round(obs.torso_roll_deg, 1),
        "torso_pitch_deg": round(obs.torso_pitch_deg, 1),
        "kick_taken": kick_taken,
        "robot_fallen": bool(obs.robot_fallen),
        "goal_scored": bool(obs.goal_scored),
    }


class ScriptedKick:
    """The 0.70 s two-leg strike, triggerable on demand.

    Same trajectory as RLRuleBasedPK's validated KICK phase: support leg
    frozen at its actual angles with a CoM-shift offset, swing leg
    knee-fold -> whip extension, early plant-down, blend back to policy.
    """

    def __init__(self, params: RLPKParams):
        self.p = params
        self.start_time: float | None = None
        self._hip0 = 0.0
        self._knee0 = 0.3
        self._left_hold: np.ndarray | None = None
        self._hip_yaw_comp = 0.0

    def begin(self, env: SoccerPKEnvRL) -> None:
        g = _shot_geometry(env, self.p)
        d = env.data
        self.start_time = g["obs"].sim_time
        self._hip0 = float(d.qpos[13])
        self._knee0 = float(d.qpos[16])
        self._left_hold = d.qpos[7:13].copy()
        self._hip_yaw_comp = float(np.clip(g["yaw_err"], -0.35, 0.35))

    def step(self, env: SoccerPKEnvRL):
        """(override_targets|None, blend) at the current sim time."""
        p = self.p
        assert self.start_time is not None
        t_k = env.data.time - self.start_time
        if t_k > p.script_time:
            return None, 1.0
        if t_k <= p.swing_time:
            hip = self._hip0 + (p.strike_hip - self._hip0) * _smoothstep(t_k / p.swing_time)
            if t_k <= 0.10:
                knee = self._knee0 + (0.80 - self._knee0) * _smoothstep(t_k / 0.10)
            else:
                knee = 0.80 + (0.02 - 0.80) * _smoothstep((t_k - 0.10) / (p.swing_time - 0.10))
            ankle = 0.10 + (-0.20 - 0.10) * _smoothstep(t_k / p.swing_time)
        elif t_k <= p.swing_time + 0.10:
            u = _smoothstep((t_k - p.swing_time) / 0.10)
            hip = p.strike_hip + (p.through_hip - p.strike_hip) * u
            knee = 0.02 + (0.25 - 0.02) * u
            ankle = -0.20
        elif t_k <= p.plant_time:
            u = _smoothstep((t_k - p.swing_time - 0.10) / (p.plant_time - p.swing_time - 0.10))
            hip = p.through_hip + (-0.15 - p.through_hip) * u
            knee = 0.25 + (0.30 - 0.25) * u
            ankle = -0.20
        else:
            hip, knee, ankle = -0.15, 0.30, -0.20
        right = np.array([hip, -0.03, self._hip_yaw_comp, knee, ankle, 0.0])
        assert self._left_hold is not None
        left = self._left_hold.copy()
        ramp = _smoothstep(t_k / 0.15) if t_k <= p.plant_time else \
            1.0 - _smoothstep((t_k - p.plant_time) / (p.script_time - p.plant_time))
        left[1] += p.shift_hip_roll * ramp
        left[5] += p.shift_ankle_roll * ramp
        if t_k <= p.plant_time:
            blend = 1.0
        else:
            blend = 1.0 - _smoothstep((t_k - p.plant_time) / (p.script_time - p.plant_time))
        return np.concatenate([left, right]), blend

    @property
    def done(self) -> bool:
        return self.start_time is None


class RLActionExecutor:
    """Executes the current action-menu choice on the RL low layer."""

    def __init__(self, env: SoccerPKEnvRL, params: RLPKParams | None = None):
        self.env = env
        self.p = params or RLPKParams()
        self.action = "HOLD"
        self.kick: ScriptedKick | None = None
        self.kick_taken = False
        self.kick_time: float | None = None
        self._recover_start: float | None = None

    def rearm_for_retry(self) -> bool:
        """Rearm only after the strike and catch-step recovery have finished."""
        if (not self.kick_taken or self.kick is not None
                or self._recover_start is None
                or self.env.data.time - self._recover_start < self.p.recover_time):
            return False
        self.kick_taken = False
        self._recover_start = None
        self.action = "GOTO_STANCE"
        return True

    def set_action(self, action: str) -> None:
        assert action in ACTIONS, f"unknown action {action}"
        if action == "KICK" and not self.kick_taken:
            self.kick = ScriptedKick(self.p)
            self.kick.begin(self.env)
            self.kick_taken = True
            self.kick_time = self.env.data.time
        self.action = action

    def step(self) -> None:
        env, p = self.env, self.p
        obs = env.get_observation()
        g = _shot_geometry(env, p)
        cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        override, blend = None, 1.0

        if self.kick is not None:
            override, blend = self.kick.step(env)
            if override is None:
                self.kick = None
                self._recover_start = obs.sim_time
            env.rl_control_step(cmd, override, blend)
            return

        if self._recover_start is not None and \
                obs.sim_time - self._recover_start < p.recover_time:
            vyaw = float(np.clip(2.2 * g["yaw_err"], -0.5, 0.5))
            roll_rad = math.radians(obs.torso_roll_deg)
            vy = float(np.clip(-1.0 * roll_rad, -0.35, 0.35))
            env.rl_control_step((p.recover_speed, vy, vyaw))
            return

        if self.action == "GOTO_STANCE":
            yaw = g["yaw"]
            c, s = math.cos(-yaw), math.sin(-yaw)
            ts = g["to_stance"]
            local = np.array([c * ts[0] - s * ts[1], s * ts[0] + c * ts[1]])
            speed = float(np.linalg.norm(local))
            cap = p.approach_cap(g["dist_to_stance"])
            if speed > cap:
                local *= cap / speed
            vyaw = float(np.clip(2.2 * g["yaw_err"], -0.7, 0.7))
            cmd = (float(local[0]), float(local[1]), vyaw)
        else:  # ALIGN or HOLD: stand, hold heading
            cmd = (0.0, 0.0, float(np.clip(2.2 * g["yaw_err"], -0.5, 0.5)))
        env.rl_control_step(cmd)


class RuleBasedDecider:
    """Reference decider using only the AI-facing state dict.

    GOTO_STANCE until close -> ALIGN until heading/rate/planted are all
    in tolerance -> KICK, repeating after a stopped shot is rearmed.
    """

    def decide(self, state: dict) -> str:
        if state["kick_taken"]:
            return "HOLD"
        if state["dist_to_stance_m"] > 0.07 or abs(state["yaw_err_deg"]) > 5.7:
            return "GOTO_STANCE"
        if not (state["both_feet_planted"] and abs(state["yaw_err_deg"]) < 2.3
                and abs(state["yaw_rate_deg_s"]) < 11.5):
            return "ALIGN"
        return "KICK"


@dataclass
class EpisodeResult:
    goal: bool = False
    fallen: bool = False
    fall_time: float | None = None
    fell_after_kick: bool = False
    kick_time: float | None = None
    kick_times: list[float] = field(default_factory=list)
    peak_ball_vx: float = 0.0
    max_ball_x: float = 0.25
    final_ball: tuple = ()
    decisions: list = field(default_factory=list)
    end_reason: str = ""
    end_time: float = 0.0


def run_ai_episode(decider, ball_lateral_speed: float = -0.11,
                   ball_offset: tuple[float, float] = (0.0, 0.0),
                   params: RLPKParams | None = None,
                   max_time: float = 30.0,
                   decision_interval: float = 0.25,
                   stop_on_fall: bool = False,
                   frame_callback: Callable[[SoccerPKEnvRL, PKStateObservation, str], None] | None = None
                   ) -> EpisodeResult:
    """Run one closed-loop episode.

    Termination: goal, timeout, or fall -- except that once a kick has
    been taken the ball is played out (crossing or stopping) even if the
    robot falls, because a kicked ball does not care about the kicker.
    Decisions pause during each shot. If the ball stops without a goal,
    an upright robot finishes recovery, reapproaches, and resumes decisions.
    The overall time limit includes every shot and API wait.
    """
    env = SoccerPKEnvRL()
    try:
        env.reset(ball_lateral_speed=ball_lateral_speed, ball_offset=ball_offset)
        params = params or RLPKParams()
        ex = RLActionExecutor(env, params)
        res = EpisodeResult()
        next_decision = 0.0
        next_frame_time = 0.0
        last_frame_time: float | None = None
        ball_still_since: float | None = None

        def observe_step() -> PKStateObservation:
            nonlocal next_frame_time, last_frame_time
            current = env.get_observation()
            if frame_callback is not None and current.sim_time + 1e-9 >= next_frame_time:
                frame_callback(env, current, ex.action)
                last_frame_time = current.sim_time
                next_frame_time += 1.0 / 30.0
            res.peak_ball_vx = max(res.peak_ball_vx, current.ball_vel[0])
            res.max_ball_x = max(res.max_ball_x, current.ball_pos[0])
            if current.robot_fallen and res.fall_time is None:
                res.fall_time = current.sim_time
                res.fallen = True
                res.fell_after_kick = bool(res.kick_times)
            return current

        steps = int(max_time / 0.002)
        for i in range(steps):
            obs = env.get_observation()
            if obs.sim_time >= max_time:
                res.end_reason = "timeout"
                break
            if not ex.kick_taken and obs.sim_time >= next_decision:
                state = dict(build_state(env, params, ex.kick_taken))
                state["time_limit_s"] = max_time
                state["remaining_time_s"] = max(0.0, max_time - obs.sim_time)
                state["kick_count"] = len(res.kick_times)
                state["current_action"] = ex.action
                state["last_api_latency_s"] = float(
                    getattr(decider, "last_latency_s", 0.0) or 0.0)
                previous = next(
                    (d for d in reversed(res.decisions) if d["applied_at"] is not None), None)
                state["previous_decision"] = ({
                    "action": previous["action"],
                    "observed_at_s": previous["sent_at"],
                    "applied_at_s": previous["applied_at"],
                    "dist_to_stance_m": previous["dist_to_stance_m"],
                    "yaw_err_deg": previous["yaw_err_deg"],
                } if previous is not None else None)
                sent_wall = datetime.now(timezone.utc).isoformat()
                t0 = time.perf_counter()
                action = decider.decide(state)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                received_wall = datetime.now(timezone.utc).isoformat()
                assert action in ACTIONS, f"decider returned {action}"
                # Latency honesty: a real (API) decider's action only lands
                # after its inference time, during which the world keeps
                # moving under the PREVIOUS action.  Deciders may report
                # `last_latency_s`; advance the sim by it before applying.
                latency_s = float(getattr(decider, "last_latency_s", 0.0) or 0.0)
                applied_at = obs.sim_time
                if latency_s > 0.0:
                    for _ in range(int(latency_s / 0.002)):
                        ex.step()
                        o2 = observe_step()
                        if o2.robot_fallen or o2.goal_scored or o2.sim_time >= max_time:
                            break
                    applied_at = env.get_observation().sim_time
                current = env.get_observation()
                can_apply = not (current.robot_fallen or current.goal_scored
                                 or current.sim_time >= max_time)
                if can_apply:
                    was_kicked = ex.kick_taken
                    ex.set_action(action)
                    if not was_kicked and ex.kick_taken:
                        res.kick_times.append(ex.kick_time)
                res.decisions.append({
                    "t": round(obs.sim_time, 2), "action": action,
                    "sent_at": obs.sim_time,
                    "received_at": obs.sim_time + latency_s,
                    "sent_wall_utc": sent_wall,
                    "received_wall_utc": received_wall,
                    "applied_at": round(applied_at, 2) if can_apply else None,
                    "latency_ms": round(latency_ms, 2),
                    "reported_latency_s": latency_s,
                    "dist_to_stance_m": state["dist_to_stance_m"],
                    "yaw_err_deg": state["yaw_err_deg"],
                    "ball_pos": state["ball_pos"],
                    "shot_number": len(res.kick_times) if ex.kick_taken else len(res.kick_times) + 1,
                    "input_state": state,
                    "reasoning_note": getattr(decider, "last_reasoning", ""),
                })
                if not can_apply:
                    res.goal = bool(current.goal_scored)
                    res.end_reason = ("goal" if current.goal_scored else
                                      "fall_before_kick" if current.robot_fallen else "timeout")
                    break
                next_decision = applied_at + decision_interval
            ex.step()
            obs = observe_step()
            if obs.goal_scored:
                res.goal = True
                res.end_reason = "goal"
                break
            if obs.robot_fallen and (stop_on_fall or not ex.kick_taken):
                res.end_reason = "fall_after_kick" if ex.kick_taken else "fall_before_kick"
                break
            if ex.kick_taken:
                speed = float(np.linalg.norm(obs.ball_vel))
                if speed < 0.08:
                    if ball_still_since is None:
                        ball_still_since = obs.sim_time
                    elif obs.sim_time - ball_still_since > 0.5:
                        if res.fallen:
                            res.end_reason = "fall_after_kick"
                            break
                        if ex.rearm_for_retry():
                            ball_still_since = None
                            next_decision = obs.sim_time
                else:
                    ball_still_since = None
                if (res.fallen and ex.kick_time is not None
                        and obs.sim_time - ex.kick_time > 8.0):
                    res.end_reason = "ball_live_timeout"
                    break
        else:
            res.end_reason = "timeout"
        obs = env.get_observation()
        if (frame_callback is not None
                and (last_frame_time is None or obs.sim_time - last_frame_time > 1e-9)):
            frame_callback(env, obs, ex.action)
        res.end_time = obs.sim_time
        res.kick_time = res.kick_times[0] if res.kick_times else None
        res.final_ball = tuple(round(v, 3) for v in obs.ball_pos)
        res.peak_ball_vx = round(res.peak_ball_vx, 2)
        res.max_ball_x = round(res.max_ball_x, 3)
        return res
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-0.11)
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()
    for t in range(args.trials):
        r = run_ai_episode(RuleBasedDecider(), args.lat)
        print(f"trial {t}: goal={r.goal} fallen={r.fallen} fall_time={r.fall_time} "
              f"kick_time={r.kick_time} peak_ball_vx={r.peak_ball_vx} "
              f"max_ball_x={r.max_ball_x} final_ball={r.final_ball} end={r.end_reason}")
        for d in r.decisions:
            print(f"   decision t={d['t']:5.2f} {d['action']:11s} "
                  f"dist={d['dist_to_stance_m']:.3f} yaw_err={d['yaw_err_deg']:+.1f} "
                  f"latency={d['latency_ms']}ms")
