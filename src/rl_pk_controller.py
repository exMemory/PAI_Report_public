"""RL-locomotion low layer + scripted strike for the soccer PK environment.

Why this file exists (see FIX_NOTES.md for the full record):
  Fault isolation on 2026-10-25 showed the hierarchical controller's baked-in
  single-leg SHIFT_WEIGHT posture falls by itself in ~1 s, while Unitree's
  official RL locomotion policy (g1_rl_locomotion.py + motion.pt) stands for
  10 s at zero command and walks stably in this same MJCF.  The AI decision
  layers (VLA / Gemini / Jev) therefore never had a working base to command.

Design (hybrid, after two measured dead ends -- see FIX_NOTES.md):
  * Approach + balance: the RL policy, driven by gentle velocity commands.
    A hip-pitch "boost" bias on top of the policy targets was tried first
    (test_rl_walk.py style): contact happened but ball speed peaked at
    ~1.4 m/s, far below the ~5 m/s an 11 m shot needs.  Gait-phase release
    timing was tried second and never fired at creep speeds.
  * The strike itself: a 0.85 s scripted right-leg trajectory (knee fold ->
    whip extension, PD gains raised toward the env's own 250/20) overrides
    ONLY the right leg.  The policy keeps the left leg and its internal
    state, and resumes full control afterwards to step and recover balance
    -- the recovery the static controller never had.

  soccer_pk_env.py is NOT modified: a subclass adds the 500 Hz RL torque
  step (mirroring step_physics' goal/fall bookkeeping), and the decision
  layer on top is a separate class.  Physics honesty is preserved: torques
  only, no state overwrites, the ball moves only through foot contact.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from soccer_pk_env import (
    SoccerPKEnv,
    GOAL_LINE_X,
    GOAL_HALF_WIDTH_Y,
    GOAL_HEIGHT_Z,
    _quat_to_roll_pitch_yaw,
    is_stable_upright,
)
from g1_rl_locomotion import (
    UnitreeG1RLPolicyNumPy,
    DEFAULT_LEG_ANGLES,
    LEG_KPS,
    LEG_KDS,
)

MOTION_PT = "assets/unitree_g1/motion.pt"
RIGHT_LEG = slice(6, 12)  # right leg entries in the 12-joint target vector

# PD gains for the scripted strike (the env's own static strike, which
# produced a 5.1 m/s ball, used kp=250 / kd=25 uniformly).
SCRIPT_KPS = np.array([250.0, 250.0, 250.0, 250.0, 200.0, 200.0])
SCRIPT_KDS = np.array([20.0, 20.0, 20.0, 20.0, 15.0, 15.0])


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class SoccerPKEnvRL(SoccerPKEnv):
    """SoccerPKEnv + Unitree RL locomotion policy as the low-level layer."""

    def __init__(self, width: int = 320, height: int = 240):
        # SoccerPKEnv.__init__ calls self.reset(), so the RL state must exist
        # before super().__init__ runs.
        self._rl_policy = UnitreeG1RLPolicyNumPy(MOTION_PT)
        self._rl_target_q = DEFAULT_LEG_ANGLES.copy()
        self._rl_last_update = -1.0
        super().__init__(width=width, height=height)
        self._actfrc = self.model.jnt_actfrcrange[1:13].copy()
        self.left_ankle_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    def reset(self, ball_lateral_speed: float = -0.04,
              ball_offset: tuple[float, float] = (0.0, 0.0)):
        obs = super().reset(ball_lateral_speed=ball_lateral_speed,
                            ball_offset=ball_offset)
        self._rl_policy.reset()
        self._rl_target_q = DEFAULT_LEG_ANGLES.copy()
        self._rl_last_update = -1.0
        return obs

    def rl_control_step(self, cmd: tuple[float, float, float],
                        override_targets: np.ndarray | None = None,
                        override_blend: float = 1.0) -> None:
        """Advance one 2 ms physics step under RL-policy torque control.

        If override_targets (12,) is given, ALL leg targets are blended
        toward it (blend 1 = full override) and driven with the
        scripted-strike PD gains; otherwise every joint follows the policy.
        Used during the strike: the support leg is frozen at its
        trigger-time policy targets while the swing leg follows the
        scripted trajectory, then control blends back to the policy.
        """
        d = self.data
        if d.time - self._rl_last_update >= 0.02 - 1e-9:
            self._rl_target_q = self._rl_policy.step_50hz(
                d.qpos[7:19], d.qvel[6:18], d.qpos[3:7], d.qvel[3:6],
                cmd, float(d.time),
            )
            self._rl_last_update = float(d.time)
        targets = self._rl_target_q
        kps, kds = LEG_KPS, LEG_KDS
        if override_targets is not None:
            targets = targets.copy()
            targets[:] = (
                (1.0 - override_blend) * targets
                + override_blend * override_targets
            )
            kps = LEG_KPS.copy()
            kds = LEG_KDS.copy()
            kps[:] = (1.0 - override_blend) * kps + override_blend * np.tile(SCRIPT_KPS, 2)
            kds[:] = (1.0 - override_blend) * kds + override_blend * np.tile(SCRIPT_KDS, 2)
        tau = (targets - d.qpos[7:19]) * kps - d.qvel[6:18] * kds
        tau = np.clip(tau, self._actfrc[:, 0], self._actfrc[:, 1])
        d.ctrl[:] = tau
        # Mirror the bookkeeping SoccerPKEnv.step_physics performs per step
        # (goal / fall flags and the kick-impulse proxy), so observations
        # and scoring semantics stay identical to the original environment.
        if float(d.qvel[self.ball_qvel_adr]) > 0.25:
            self._kick_impulse_applied = True
        mujoco.mj_step(self.model, d)
        bx, by, bz = d.xpos[self.ball_body_id]
        if bx >= GOAL_LINE_X and abs(by) <= GOAL_HALF_WIDTH_Y and bz <= GOAL_HEIGHT_Z:
            self.has_scored_goal = True
        pz = float(d.xpos[self.pelvis_body_id][2])
        quat = d.qpos[self.base_qpos_adr + 3: self.base_qpos_adr + 7]
        roll_rad, pitch_rad, _ = _quat_to_roll_pitch_yaw(quat)
        roll_deg, pitch_deg = float(np.rad2deg(roll_rad)), float(np.rad2deg(pitch_rad))
        angvel = d.qvel[self.base_qvel_adr + 3: self.base_qvel_adr + 6]
        if pz < 0.52 or abs(roll_deg) > 42.0 or abs(pitch_deg) > 42.0:
            self.has_fallen = True
            self.recovery_mode = True
            self.kicking_suppressed = True
        elif is_stable_upright(pz, roll_deg, pitch_deg, float(angvel[0]), float(angvel[1])):
            self.has_fallen = False
            if self.current_recovery_decision is None:
                self.recovery_mode = False
                self.kicking_suppressed = False


@dataclass
class RLPKParams:
    stance_dist: float = 0.36      # pelvis distance behind the ball at kick time
    stance_lateral: float = 0.095  # pelvis left of ball line so ball sits on right-foot line
    max_speed: float = 0.35        # approach speed cap near the ball (m/s) -- gentle on purpose
    far_speed: float = 0.9         # cruise cap when far from the stance (m/s)
    near_dist: float = 0.4         # at/below this distance the gentle cap applies (m)
    far_dist: float = 1.6          # at/above this distance the cruise cap applies (m)
    settle_time: float = 0.35      # minimum zero-command settle before the strike (s)
    align_pos_tol: float = 0.05    # stance tolerance to enter SETTLE (m)
    align_yaw_tol: float = 0.06    # heading tolerance to enter SETTLE (rad)
    settle_yaw_tol: float = 0.04   # heading tolerance to release the strike (rad)
    settle_rate_tol: float = 0.20  # yaw-rate tolerance to release the strike (rad/s)
    settle_hold: float = 0.10     # continuous in-tolerance time before release (s)
    swing_time: float = 0.20       # trigger -> ball-contact window (s)
    aim_bias_deg: float = 12.0      # calibrated aim bias added to the shot direction (deg)
    script_time: float = 0.70      # total override incl. blend-out (s)
    strike_hip: float = -2.35      # hip-pitch demand through contact (rad)
    through_hip: float = -2.10     # hip pitch at follow-through end (rad)
    recover_time: float = 0.9      # post-kick catch-step duration (s)
    recover_speed: float = 0.28    # forward step speed while recovering (m/s)
    plant_time: float = 0.45       # swing leg planted back down by this script time (s)
    shift_hip_roll: float = 0.22   # support hip-roll offset: CoM over the support foot (rad)
    shift_ankle_roll: float = -0.18  # support ankle-roll offset paired with it (rad)

    def approach_cap(self, dist_to_stance: float) -> float:
        """Distance-blended approach speed cap.

        The policy walks stably well above the gentle near-ball cap
        (measured: no falls at commands up to 1.1 m/s in this MJCF), but
        its stop overshoot grows with speed (~0.2 m at 0.5 m/s, ~0.44 m
        at 1.1 m/s), so cruise fast only while far from the stance and
        blend down to the gentle cap for the final approach.
        """
        if dist_to_stance <= self.near_dist:
            return self.max_speed
        if dist_to_stance >= self.far_dist:
            return self.far_speed
        u = (dist_to_stance - self.near_dist) / (self.far_dist - self.near_dist)
        u = u * u * (3.0 - 2.0 * u)
        return self.max_speed + (self.far_speed - self.max_speed) * u


class RLRuleBasedPK:
    """Rule-based decision layer: RL approach, scripted strike.

    Phases: APPROACH (walk to the stance point behind the ball) -> SETTLE
    (zero command, let the policy plant both feet) -> KICK (scripted
    right-leg trajectory) -> DONE (policy stands / recovers).
    The stance point tracks the live ball position every decision tick, so
    a slowly rolling ball (nonzero lateral speed) is still intercepted.
    """

    def __init__(self, params: RLPKParams | None = None):
        self.p = params or RLPKParams()
        self.phase = "APPROACH"
        self._phase_start: float | None = None
        self.kick_time: float | None = None
        self._hip0 = 0.0
        self._knee0 = 0.3
        self._left_hold: np.ndarray | None = None
        self._hip_yaw_comp = 0.0
        self._settle_ok_since: float | None = None

    def _script_targets(self, t_k: float, env: SoccerPKEnvRL) -> np.ndarray | None:
        """Full 12-joint override at script time t_k; None once it ends.

        Left leg: frozen at its trigger-time policy targets (a planted,
        rigid support like the original static strike).  Right leg: knee
        folds, then whips open while the hip drives to strike_hip -- the
        demand stays beyond the contact point so the leg is still
        accelerating when it meets the ball.
        """
        p = self.p
        if t_k > p.script_time:
            return None
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
            # Plant the swing leg back down on its own side: the measured
            # failure is a sideways (roll) fall, and the natural catch is
            # putting the kicking leg down, not holding it up.
            u = _smoothstep((t_k - p.swing_time - 0.10) / (p.plant_time - p.swing_time - 0.10))
            hip = p.through_hip + (-0.15 - p.through_hip) * u
            knee = 0.25 + (0.30 - 0.25) * u
            ankle = -0.20
        else:
            hip, knee, ankle = -0.15, 0.30, -0.20  # default stance, hold
        right = np.array([hip, -0.03, self._hip_yaw_comp, knee, ankle, 0.0])
        assert self._left_hold is not None
        left = self._left_hold.copy()
        # CoM shift over the support foot during single support (the same
        # idea as the env's SHIFT_WEIGHT pose): without it the pelvis sits
        # ~11 cm medial to the only supporting foot and topples inward --
        # measured as a monotonic roll fall even after both feet replant.
        ramp = _smoothstep(t_k / 0.15) if t_k <= p.plant_time else \
            1.0 - _smoothstep((t_k - p.plant_time) / (p.script_time - p.plant_time))
        left[1] += p.shift_hip_roll * ramp
        left[5] += p.shift_ankle_roll * ramp
        return np.concatenate([left, right])

    def _script_blend(self, t_k: float) -> float:
        p = self.p
        if t_k <= p.plant_time:
            return 1.0
        return 1.0 - _smoothstep((t_k - p.plant_time) / (p.script_time - p.plant_time))

    def decide(self, env: SoccerPKEnvRL):
        """Return (cmd, right_leg_targets|None, blend) for this tick."""
        obs = env.get_observation()
        d = env.data
        ball = np.array(obs.ball_pos[:2])
        pelvis = np.array(obs.pelvis_pos[:2])
        yaw = _yaw_from_quat(*d.qpos[3:7])

        shot_yaw = math.atan2(0.0 - ball[1], GOAL_LINE_X - ball[0]) \
            + math.radians(self.p.aim_bias_deg)
        shot_dir = np.array([math.cos(shot_yaw), math.sin(shot_yaw)])
        perp = np.array([-shot_dir[1], shot_dir[0]])
        stance = ball - shot_dir * self.p.stance_dist + perp * self.p.stance_lateral
        to_stance = stance - pelvis
        dist = float(np.linalg.norm(to_stance))
        yaw_err = _wrap_angle(shot_yaw - yaw)

        if self.phase == "APPROACH":
            if dist < self.p.align_pos_tol and abs(yaw_err) < self.p.align_yaw_tol:
                self.phase = "SETTLE"
                self._phase_start = obs.sim_time
                return (0.0, 0.0, 0.0), None, 1.0
            c, s = math.cos(-yaw), math.sin(-yaw)
            local = np.array([c * to_stance[0] - s * to_stance[1],
                              s * to_stance[0] + c * to_stance[1]])
            speed = float(np.linalg.norm(local))
            if speed > self.p.max_speed:
                local *= self.p.max_speed / speed
            vyaw = float(np.clip(2.2 * yaw_err, -0.7, 0.7))
            return (float(local[0]), float(local[1]), vyaw), None, 1.0

        if self.phase == "SETTLE":
            assert self._phase_start is not None
            # Heading-hold while settling; release the strike only after the
            # heading AND yaw rate stay inside tight tolerances for a
            # continuous hold -- a spinning base drags the shot sideways.
            vyaw = float(np.clip(2.2 * yaw_err, -0.5, 0.5))
            yaw_rate = float(d.qvel[5])
            settled_long_enough = obs.sim_time - self._phase_start >= self.p.settle_time
            # Both feet must actually be planted -- releasing mid-gait with
            # the support foot airborne dooms the strike regardless of the
            # swing (measured: consistent roll fall ~0.7 s after release).
            both_planted = float(d.xpos[env.left_ankle_body][2]) < 0.055 \
                and float(d.xpos[env.right_ankle_body][2]) < 0.055
            if abs(yaw_err) < self.p.settle_yaw_tol and abs(yaw_rate) < self.p.settle_rate_tol \
                    and both_planted:
                if self._settle_ok_since is None:
                    self._settle_ok_since = obs.sim_time
            else:
                self._settle_ok_since = None
            hold_done = self._settle_ok_since is not None \
                and obs.sim_time - self._settle_ok_since >= self.p.settle_hold
            # Best-effort fallback: never wait forever for a perfect window.
            timed_out = obs.sim_time - self._phase_start >= 2.5 and abs(yaw_err) < 0.08
            if settled_long_enough and (hold_done or timed_out):
                self.phase = "KICK"
                self.kick_time = obs.sim_time
                self._hip0 = float(d.qpos[13])
                self._knee0 = float(d.qpos[16])
                # Freeze the support leg at its ACTUAL planted angles, not
                # at policy targets that may describe a mid-step pose.
                self._left_hold = d.qpos[7:13].copy()
                # Aim the swing plane at the goal, not at the (mis)facing:
                # compensate the residual heading error via hip yaw.
                self._hip_yaw_comp = float(np.clip(_wrap_angle(shot_yaw - yaw), -0.35, 0.35))
            return (0.0, 0.0, vyaw), None, 1.0

        if self.phase == "KICK":
            assert self.kick_time is not None
            t_k = obs.sim_time - self.kick_time
            targets = self._script_targets(t_k, env)
            if targets is None:
                self.phase = "RECOVER"
                self._phase_start = obs.sim_time
                return (0.0, 0.0, 0.0), None, 1.0
            return (0.0, 0.0, 0.0), targets, self._script_blend(t_k)

        if self.phase == "RECOVER":
            # Step forward with the policy to catch the follow-through
            # momentum, heading held, then stand.
            assert self._phase_start is not None
            if obs.sim_time - self._phase_start >= self.p.recover_time:
                self.phase = "DONE"
                return (0.0, 0.0, 0.0), None, 1.0
            vyaw = float(np.clip(2.2 * yaw_err, -0.5, 0.5))
            # Lateral catch step toward the tipping side (the measured
            # post-kick failure is a roll fall toward the kicking-leg side).
            roll_rad = math.radians(obs.torso_roll_deg)
            vy = float(np.clip(-1.0 * roll_rad, -0.35, 0.35))
            return (self.p.recover_speed, vy, vyaw), None, 1.0

        return (0.0, 0.0, 0.0), None, 1.0


def run_episode(ball_lateral_speed: float = -0.11,
                params: RLPKParams | None = None,
                max_time: float = 15.0) -> dict:
    env = SoccerPKEnvRL()
    try:
        env.reset(ball_lateral_speed=ball_lateral_speed)
        ctl = RLRuleBasedPK(params)
        cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)
        targets: np.ndarray | None = None
        blend = 1.0
        peak_ball_vx = 0.0
        max_ball_x = 0.25
        fall_time = None
        steps = int(max_time / 0.002)
        for i in range(steps):
            if i % 17 == 0:  # decision tick ~29 Hz
                cmd, targets, blend = ctl.decide(env)
            env.rl_control_step(cmd, targets, blend)
            obs = env.get_observation()
            peak_ball_vx = max(peak_ball_vx, obs.ball_vel[0])
            max_ball_x = max(max_ball_x, obs.ball_pos[0])
            if fall_time is None and obs.robot_fallen:
                fall_time = obs.sim_time
            if obs.goal_scored:
                break
        obs = env.get_observation()
        return {
            "goal": bool(obs.goal_scored),
            "fallen": bool(obs.robot_fallen),
            "fall_time": fall_time,
            "kick_time": ctl.kick_time,
            "peak_ball_vx": round(peak_ball_vx, 2),
            "max_ball_x": round(max_ball_x, 3),
            "final_ball": tuple(round(v, 3) for v in obs.ball_pos),
            "final_pelvis_z": round(obs.pelvis_pos[2], 3),
            "end_phase": ctl.phase,
        }
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-0.11)
    ap.add_argument("--stance", type=float, default=0.36)
    ap.add_argument("--lateral", type=float, default=0.095)
    ap.add_argument("--strike-hip", type=float, default=-2.35)
    ap.add_argument("--aim-bias", type=float, default=12.0)
    ap.add_argument("--swing-time", type=float, default=0.20)
    ap.add_argument("--script-time", type=float, default=0.70)
    ap.add_argument("--plant-time", type=float, default=0.45)
    ap.add_argument("--through-hip", type=float, default=-2.10)
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()
    params = RLPKParams(stance_dist=args.stance, stance_lateral=args.lateral,
                        strike_hip=args.strike_hip, aim_bias_deg=args.aim_bias,
                        swing_time=args.swing_time, script_time=args.script_time,
                        through_hip=args.through_hip, plant_time=args.plant_time)
    for t in range(args.trials):
        print(f"trial {t}: {run_episode(args.lat, params)}")
