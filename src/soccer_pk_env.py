"""
MuJoCo Soccer Penalty Kick (PK) Environment for the Unitree G1 12-leg-DoF model.

100% Pure Rigid-Body Physics (Zero Cheats / No External Forces / No Velocity Teleportation):
1. Unitree G1 mesh model with 12 actuated leg joints (`assets/unitree_g1/g1_soccer_pk.xml`).
2. True 6-DoF `<freejoint name="floating_base_joint"/>` under `9.81 m/s^2` gravity.
   - NO `xfrc_applied` external pelvis wrench is ever applied. The robot stands and kicks
     solely through its 12 leg joint actuators (`data.ctrl`) and foot-ground contact friction.
3. True Rigid-Body Foot-to-Ball Contact (`contype="1" conaffinity="1"`).
   - NO `data.qvel` ball velocity overwrite is ever used. The ball moves strictly when
     struck by the Unitree G1's 3D foot geometry (`right_foot_instep` / mesh) via MuJoCo's
     contact solver.
4. Classic black-and-white soccer ball on the 11.0-meter FIFA Penalty Spot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
G1_SOCCER_XML_PATH = PROJECT_ROOT / "assets" / "unitree_g1" / "g1_soccer_pk.xml"

# Official FIFA 11.0m PK Geometry (7.32m x 2.44m Goal = 3:1 Aspect Ratio)
PENALTY_SPOT_X = 0.25
GOAL_LINE_X = 11.25         # Exactly 11.00m from the penalty spot (0.25m -> 11.25m)
GOAL_HALF_WIDTH_Y = 3.60    # FIFA Official Goal posts at Y = +/- 3.66m (Width = 7.32m)
GOAL_HEIGHT_Z = 2.40        # FIFA Official Crossbar at Z = 2.44m (Height = 2.44m)

ALLOWED_FALL_ORIENTATIONS: Tuple[str, ...] = (
    "UPRIGHT",
    "PRONE",
    "SUPINE",
    "LEFT_LATERAL",
    "RIGHT_LATERAL",
)

ALLOWED_RECOVERY_STAGES: Tuple[str, ...] = (
    "RECOVERY_ORIENT",
    "RECOVERY_TUCK",
    "RECOVERY_PLANT",
    "RECOVERY_EXTEND",
    "RECOVERY_BALANCE",
    "RECOVERY_HOLD",
)

RECOVERY_DELTA_BOUNDS = {
    "hip_pitch_delta_rad": (-0.5, 0.5),
    "knee_delta_rad": (-0.5, 0.5),
    "ankle_pitch_delta_rad": (-0.5, 0.5),
}

SUPINE_RECOVERY_KEYFRAMES = {
    "RECOVERY_ORIENT": (2.252625941412, 1.998702272088, -0.645772619240, 1.299852534233, 0.133185022790, 0.135154933946, 0.860368781266, -1.433963798372, -1.636880576229, 2.255498017264, 0.320796048816, 0.162256774225),
    "RECOVERY_TUCK": (-0.804270622633, 1.103258269479, -1.851241190022, 1.446959644793, -0.702875605870, 0.124081242185, -1.999746210025, -1.326388105202, 1.603701933538, 1.548477992104, 0.012223054772, -0.043115279989),
    "RECOVERY_PLANT": (-1.487441983823, 1.646365114034, -2.544080884423, 2.597061968374, -0.059405973374, 0.227952072788, -0.434934186025, -0.757084689166, 2.031522025812, 0.927369149764, 0.446284489130, 0.087186017602),
}


@dataclass
class RecoveryDecision:
    """Separate control contract for fall recovery (strictly distinct from PK kick phases)."""
    recovery_stage: str                  # Must be one of ALLOWED_RECOVERY_STAGES
    hip_pitch_delta_rad: float = 0.0     # Bounded joint target adjustment [-0.5, 0.5]
    knee_delta_rad: float = 0.0          # Bounded joint target adjustment [-0.5, 0.5]
    ankle_pitch_delta_rad: float = 0.0   # Bounded joint target adjustment [-0.5, 0.5]
    confidence: float = 1.0              # 0.0 to 1.0
    inference_ms: float = 0.0            # Latency in ms
    reasoning_note: str = ""


@dataclass
class PKStateObservation:
    """Structured state passed to the AI models (#1 Pure VLA, #2 Gemini ER 2, #3 Jev)."""
    sim_time: float
    ball_pos: Tuple[float, float, float]
    ball_vel: Tuple[float, float, float]
    foot_pos: Tuple[float, float, float]
    pelvis_pos: Tuple[float, float, float]
    torso_roll_deg: float
    torso_pitch_deg: float
    torso_roll_rate_rad_s: float
    torso_pitch_rate_rad_s: float
    support_foot_pos: Tuple[float, float, float]
    support_hip_pitch_rad: float
    support_hip_roll_rad: float
    support_knee_rad: float
    current_phase: str
    ball_in_strike_zone: bool
    robot_fallen: bool
    ball_distance_m: float
    goal_scored: bool
    ball_missed: bool
    fall_orientation: str = "UPRIGHT"
    is_stable_upright: bool = True
    current_recovery_stage: Optional[str] = None
    recovery_stage_elapsed_sec: float = 0.0


@dataclass
class AIControlDecision:
    """Standardized control output produced by Pure VLA (#1), Gemini ER 2 (#2), or Jev (#3)."""
    action_phase: str          # "SHIFT_WEIGHT", "WINDUP_LEG", "KICK_STRIKE", "FOLLOW_THROUGH"
    aim_yaw_deg: float         # Target kick yaw angle (-18.0 to +18.0 deg)
    kick_power: float          # 0.4 to 1.25
    confidence: float          # 0.0 to 1.0
    inference_ms: float        # Inference latency in ms
    raw_vla_joint_mode: bool = False  # True for direct VLA-style leg joint action
    detected_ball_uv: Optional[Tuple[float, float]] = None
    reasoning_note: str = ""
    support_hip_pitch_delta_rad: float = 0.0
    support_hip_roll_delta_rad: float = 0.0
    support_knee_delta_rad: float = 0.0


def _bounded_posture_delta(value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return 0.0
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return 0.0
    return float(np.clip(numeric_value, minimum, maximum))


def _quat_to_roll_pitch_yaw(quat_wxyz: np.ndarray) -> Tuple[float, float, float]:
    """Convert MuJoCo quaternion (w, x, y, z) to (roll, pitch, yaw) in radians."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def classify_fall_orientation(
    pelvis_z: float,
    torso_roll_deg: float,
    torso_pitch_deg: float,
) -> str:
    """
    Classify humanoid fall orientation from pelvis elevation and torso orientation.
    Returns one of ALLOWED_FALL_ORIENTATIONS: 'UPRIGHT', 'PRONE', 'SUPINE', 'LEFT_LATERAL', 'RIGHT_LATERAL'.
    """
    if pelvis_z >= 0.70 and abs(torso_roll_deg) <= 15.0 and abs(torso_pitch_deg) <= 15.0:
        return "UPRIGHT"
    # Near the Euler pitch singularity, roll can also approach 90 degrees.
    if abs(torso_pitch_deg) >= 60.0:
        return "SUPINE" if torso_pitch_deg > 0.0 else "PRONE"
    if abs(torso_roll_deg) >= abs(torso_pitch_deg):
        return "RIGHT_LATERAL" if torso_roll_deg > 0.0 else "LEFT_LATERAL"
    return "SUPINE" if torso_pitch_deg > 0.0 else "PRONE"


def is_stable_upright(
    pelvis_z: float,
    torso_roll_deg: float,
    torso_pitch_deg: float,
    torso_roll_rate_rad_s: float = 0.0,
    torso_pitch_rate_rad_s: float = 0.0,
    min_pelvis_z: float = 0.68,
    max_tilt_deg: float = 18.0,
    max_rate_rad_s: float = 1.2,
) -> bool:
    """
    Check strict observation-based criteria for stable upright standing:
    pelvis elevated, small roll/pitch deviation, and low angular velocities.
    """
    return bool(
        pelvis_z >= min_pelvis_z
        and abs(torso_roll_deg) <= max_tilt_deg
        and abs(torso_pitch_deg) <= max_tilt_deg
        and abs(torso_roll_rate_rad_s) <= max_rate_rad_s
        and abs(torso_pitch_rate_rad_s) <= max_rate_rad_s
    )


class SoccerPKEnv:
    """
    MuJoCo environment for the Unitree G1 12-leg-DoF Single-Leg 11.0m Penalty Kick benchmark
    using 100% pure rigid-body dynamics and actuator torques only.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        if not G1_SOCCER_XML_PATH.exists():
            raise FileNotFoundError(f"Unitree G1 MJCF not found at: {G1_SOCCER_XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(G1_SOCCER_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Cache body & joint IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.right_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        self.left_foot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )

        self.base_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        self.base_qpos_adr = int(self.model.jnt_qposadr[self.base_jnt_id])
        self.base_qvel_adr = int(self.model.jnt_dofadr[self.base_jnt_id])

        self.ball_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_jnt_id])
        self.ball_qvel_adr = int(self.model.jnt_dofadr[self.ball_jnt_id])

        # First actuated robot joint starts immediately after the 7-DoF floating_base_joint
        self.robot_qpos_adr = self.base_qpos_adr + 7
        self.robot_qvel_adr = self.base_qvel_adr + 6

        self.current_phase = "SHIFT_WEIGHT"
        self.current_decision: Optional[AIControlDecision] = None
        self.current_recovery_decision: Optional[RecoveryDecision] = None
        self.current_recovery_stage: Optional[str] = None
        self.recovery_stage_started_at: Optional[float] = None
        self.recovery_initial_orientation: Optional[str] = None
        self.has_scored_goal = False
        self.has_fallen = False
        self.recovery_mode = False
        self.kicking_suppressed = False
        self._kick_impulse_applied = False
        self.joint_targets = np.zeros(self.model.nu, dtype=np.float64)
        self._actuator_joint_ids = self.model.actuator_trnid[:, 0].astype(np.intp)
        self._actuator_qpos_adr = self.model.jnt_qposadr[self._actuator_joint_ids]
        self._actuator_qvel_adr = self.model.jnt_dofadr[self._actuator_joint_ids]
        joint_force_ranges = self.model.jnt_actfrcrange[self._actuator_joint_ids]
        self._actuator_torque_limits = np.max(np.abs(joint_force_ranges), axis=1)
        self._actuator_torque_limits[self._actuator_torque_limits == 0.0] = 100.0
        self.joint_kp = 250.0
        self.joint_kd = 25.0
        self.reset()

    def _default_stand_ctrl(self) -> np.ndarray:
        """
        Default standing joint angles for the 12 leg actuators of Unitree G1:
          0..5:   left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
          6..11:  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
        """
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        # Slight knee bend standing posture
        ctrl[0] = -0.15   # left_hip_pitch
        ctrl[3] = 0.30    # left_knee
        ctrl[4] = -0.15   # left_ankle_pitch
        ctrl[6] = -0.15   # right_hip_pitch
        ctrl[9] = 0.30    # right_knee
        ctrl[10] = -0.15  # right_ankle_pitch
        return ctrl

    def reset(self, ball_lateral_speed: float = -0.04,
              ball_offset: tuple[float, float] = (0.0, 0.0)) -> PKStateObservation:
        """
        Reset the Unitree G1 on its feet under 9.81 m/s^2 gravity and place the black-and-white
        soccer ball at the penalty spot (X = 0.22m, 11.00m from the goal line at X = 11.22m).
        """
        mujoco.mj_resetData(self.model, self.data)
        self.current_phase = "SHIFT_WEIGHT"
        self.current_decision = None
        self.current_recovery_decision = None
        self.current_recovery_stage = None
        self.recovery_stage_started_at = None
        self.recovery_initial_orientation = None
        self.has_scored_goal = False
        self.has_fallen = False
        self.recovery_mode = False
        self.kicking_suppressed = False
        self._kick_impulse_applied = False

        # Initialize Unitree G1 standing on the pitch (pelvis at Z = 0.782m)
        stand_ctrl = self._default_stand_ctrl()
        self.joint_targets = stand_ctrl.copy()
        self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 7] = [
            0.0, 0.0, 0.782, 1.0, 0.0, 0.0, 0.0
        ]
        self.data.qpos[self.robot_qpos_adr : self.robot_qpos_adr + self.model.nu] = stand_ctrl
        self.data.ctrl[:] = 0.0

        # Place ball in front of the right kicking foot (X = 0.25m, Y = -0.095m) with a 2cm clearance from the standing toe
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7] = [
            PENALTY_SPOT_X + ball_offset[0], -0.095 + ball_offset[1], 0.103,
            1.0, 0.0, 0.0, 0.0
        ]
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = [
            0.0, ball_lateral_speed, 0.0, 0.0, 0.0, 0.0
        ]

        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> PKStateObservation:
        ball_pos = tuple(float(x) for x in self.data.xpos[self.ball_body_id])
        foot_pos = tuple(float(x) for x in self.data.xpos[self.right_foot_body_id])
        pelvis_pos = tuple(float(x) for x in self.data.xpos[self.pelvis_body_id])

        ball_vel = tuple(
            float(x) for x in self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3]
        )

        pelvis_quat = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]
        roll_rad, pitch_rad, _ = _quat_to_roll_pitch_yaw(pelvis_quat)
        pelvis_angvel = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]
        roll_deg = float(np.rad2deg(roll_rad))
        pitch_deg = float(np.rad2deg(pitch_rad))
        support_foot_pos = tuple(float(x) for x in self.data.xpos[self.left_foot_body_id])

        # Check if robot lost balance and fell under gravity, or recovered to stable upright
        pelvis_z = pelvis_pos[2]
        roll_rate = float(pelvis_angvel[0])
        pitch_rate = float(pelvis_angvel[1])

        stable_upright = is_stable_upright(
            pelvis_z=pelvis_z,
            torso_roll_deg=roll_deg,
            torso_pitch_deg=pitch_deg,
            torso_roll_rate_rad_s=roll_rate,
            torso_pitch_rate_rad_s=pitch_rate,
        )

        fall_orientation = classify_fall_orientation(
            pelvis_z=pelvis_z,
            torso_roll_deg=roll_deg,
            torso_pitch_deg=pitch_deg,
        )

        if pelvis_z < 0.52 or abs(roll_deg) > 42.0 or abs(pitch_deg) > 42.0:
            if self.recovery_initial_orientation is None:
                self.recovery_initial_orientation = fall_orientation
            self.has_fallen = True
            self.recovery_mode = True
            self.kicking_suppressed = True
        elif stable_upright and self.has_fallen:
            self.has_fallen = False
            self.recovery_mode = False
            self.kicking_suppressed = False
            self.recovery_initial_orientation = None

        # Strike zone in front of G1's right foot
        in_zone = (0.18 <= ball_pos[0] <= 0.36) and (-0.155 <= ball_pos[1] <= -0.085)

        # Official 11.0m Goal check: ball crosses X >= 11.22m within posts (|Y| <= 3.60m, Z <= 2.40m)
        if (
            ball_pos[0] >= GOAL_LINE_X
            and abs(ball_pos[1]) <= GOAL_HALF_WIDTH_Y
            and ball_pos[2] <= GOAL_HEIGHT_Z
        ):
            self.has_scored_goal = True

        ball_dist_m = max(0.0, float(ball_pos[0] - PENALTY_SPOT_X))
        ball_speed_xy = math.hypot(ball_vel[0], ball_vel[1])

        ball_missed = (not self.has_scored_goal) and (
            (ball_pos[0] >= GOAL_LINE_X and abs(ball_pos[1]) > GOAL_HALF_WIDTH_Y)
            or self.has_fallen
            or (self.data.time > 1.6 and ball_pos[0] < 11.0 and ball_speed_xy < 0.15)
            or (self.data.time > 2.8)
        )

        return PKStateObservation(
            sim_time=float(self.data.time),
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            foot_pos=foot_pos,
            pelvis_pos=pelvis_pos,
            torso_roll_deg=roll_deg,
            torso_pitch_deg=pitch_deg,
            torso_roll_rate_rad_s=float(pelvis_angvel[0]),
            torso_pitch_rate_rad_s=float(pelvis_angvel[1]),
            support_foot_pos=support_foot_pos,
            support_hip_pitch_rad=float(self.data.qpos[self._actuator_qpos_adr[0]]),
            support_hip_roll_rad=float(self.data.qpos[self._actuator_qpos_adr[1]]),
            support_knee_rad=float(self.data.qpos[self._actuator_qpos_adr[3]]),
            current_phase=self.current_phase,
            ball_in_strike_zone=in_zone,
            robot_fallen=self.has_fallen,
            ball_distance_m=ball_dist_m,
            goal_scored=self.has_scored_goal,
            ball_missed=ball_missed,
            fall_orientation=fall_orientation,
            is_stable_upright=stable_upright,
            current_recovery_stage=self.current_recovery_stage,
            recovery_stage_elapsed_sec=(
                max(0.0, self.data.time - self.recovery_stage_started_at)
                if self.recovery_stage_started_at is not None
                else 0.0
            ),
        )

    def apply_decision(self, decision: AIControlDecision) -> None:
        """
        Convert an AI decision into 12 leg-joint targets for the low-level PD controller.
        NO external forces (`xfrc_applied`) or velocity hacks are used anywhere.
        If fallen or in recovery mode, kicking is suppressed until upright.
        """
        self.current_decision = decision
        self.current_phase = decision.action_phase

        if self.recovery_mode or self.has_fallen:
            # Trigger recovery on fall and suppress kicking until upright
            self.kicking_suppressed = True
            self.joint_targets = self._default_stand_ctrl()
            return

        phase = decision.action_phase
        yaw_rad = float(np.deg2rad(np.clip(decision.aim_yaw_deg, -20.0, 20.0)))
        power = float(np.clip(decision.kick_power, 0.4, 1.25))

        ctrl = self._default_stand_ctrl()

        if decision.raw_vla_joint_mode:
            # #1 LEFT SCREEN: PURE VLA (Direct 12-DoF Joint Action without CoM shift compensation)
            if phase == "SHIFT_WEIGHT":
                ctrl[0] = -0.15
                ctrl[1] = 0.02
                ctrl[3] = 0.30
                ctrl[4] = -0.15
                ctrl[6] = -0.30
                ctrl[9] = 0.65
            elif phase == "WINDUP_LEG":
                ctrl[0] = -0.20
                ctrl[1] = 0.02
                ctrl[3] = 0.32
                ctrl[4] = -0.16
                ctrl[6] = 0.45
                ctrl[7] = -0.08
                ctrl[8] = yaw_rad
                ctrl[9] = 1.15
                ctrl[10] = 0.10
            elif phase == "KICK_STRIKE":
                ctrl[0] = -0.22
                ctrl[1] = 0.00
                ctrl[3] = 0.30
                ctrl[4] = -0.18
                ctrl[6] = -0.95 * power
                ctrl[7] = -0.08
                ctrl[8] = yaw_rad
                ctrl[9] = 0.05
                ctrl[10] = -0.20
            else:
                ctrl[0] = -0.20
                ctrl[1] = 0.00
                ctrl[3] = 0.30
                ctrl[4] = -0.15
                ctrl[6] = -0.55
                ctrl[9] = 0.25
        else:
            # HIERARCHICAL JOINT CONTROLLER (Uses only 12 leg joint actuators: shifts pelvis over left leg via hip/ankle roll)
            ctrl[0] = -0.18   # left_hip_pitch
            ctrl[1] = 0.26    # left_hip_roll (shift CoM toward left support foot via actuator torque)
            ctrl[3] = 0.36    # left_knee
            ctrl[4] = -0.18   # left_ankle_pitch
            ctrl[5] = -0.24   # left_ankle_roll

            if phase == "SHIFT_WEIGHT":
                ctrl[8] = yaw_rad
                ctrl[6] = -0.10
                ctrl[9] = 0.45
            elif phase == "WINDUP_LEG":
                ctrl[8] = yaw_rad
                ctrl[6] = 0.55    # right_hip_pitch backswing
                ctrl[7] = -0.05   # right_hip_roll
                ctrl[9] = 1.25    # right_knee cocked
                ctrl[10] = 0.15   # right_ankle_pitch
            elif phase == "KICK_STRIKE":
                ctrl[8] = yaw_rad
                ctrl[6] = -1.92 * min(power, 1.2)  # forward hip swing via joint actuator
                ctrl[7] = -0.05
                ctrl[9] = 0.02                     # knee extension via joint actuator
                ctrl[10] = -0.20                   # firm ankle
            elif phase == "FOLLOW_THROUGH":
                ctrl[8] = yaw_rad
                ctrl[6] = -0.55
                ctrl[7] = -0.05
                ctrl[9] = 0.25
                ctrl[10] = -0.10

            ctrl[0] += _bounded_posture_delta(
                decision.support_hip_pitch_delta_rad, -0.4, 0.4
            )
            ctrl[1] += _bounded_posture_delta(
                decision.support_hip_roll_delta_rad, -0.15, 0.15
            )
            ctrl[3] += _bounded_posture_delta(
                decision.support_knee_delta_rad, -0.3, 0.45
            )

        self.joint_targets = ctrl

    def apply_recovery_decision(self, decision: RecoveryDecision) -> None:
        """
        Deterministic bounded PD/keyframe executor for recovery using only the 12 existing actuators.
        Strictly zero external forces (`xfrc_applied`) and zero `qpos`/`qvel` teleportation.
        """
        if decision.recovery_stage not in ALLOWED_RECOVERY_STAGES:
            raise ValueError(
                f"Invalid recovery_stage '{decision.recovery_stage}'. Must be one of {ALLOWED_RECOVERY_STAGES}."
            )

        self.current_recovery_decision = decision
        if decision.recovery_stage != self.current_recovery_stage:
            self.recovery_stage_started_at = float(self.data.time)
        self.current_recovery_stage = decision.recovery_stage
        stage = decision.recovery_stage

        hp_delta = _bounded_posture_delta(decision.hip_pitch_delta_rad, -0.5, 0.5)
        k_delta = _bounded_posture_delta(decision.knee_delta_rad, -0.5, 0.5)
        ap_delta = _bounded_posture_delta(decision.ankle_pitch_delta_rad, -0.5, 0.5)

        ctrl = np.zeros(self.model.nu, dtype=np.float64)

        observation = self.get_observation()
        self.recovery_mode = True
        self.kicking_suppressed = True
        recovery_orientation = self.recovery_initial_orientation or observation.fall_orientation
        if recovery_orientation == "SUPINE" and stage in SUPINE_RECOVERY_KEYFRAMES:
            ctrl[:] = SUPINE_RECOVERY_KEYFRAMES[stage]
            self.joint_targets = ctrl
            return

        if stage == "RECOVERY_ORIENT":
            # Orient: align legs symmetrically to prepare for tuck
            ctrl[0] = -0.20 + hp_delta
            ctrl[3] = 0.40 + k_delta
            ctrl[4] = -0.20 + ap_delta
            ctrl[6] = -0.20 + hp_delta
            ctrl[9] = 0.40 + k_delta
            ctrl[10] = -0.20 + ap_delta
        elif stage == "RECOVERY_TUCK":
            # Tuck: flex hips and bend knees close to pelvis
            ctrl[0] = -0.80 + hp_delta
            ctrl[3] = 1.30 + k_delta
            ctrl[4] = -0.40 + ap_delta
            ctrl[6] = -0.80 + hp_delta
            ctrl[9] = 1.30 + k_delta
            ctrl[10] = -0.40 + ap_delta
        elif stage == "RECOVERY_PLANT":
            # Plant: position feet downward under pelvis with slightly widened stance
            ctrl[0] = -0.50 + hp_delta
            ctrl[1] = 0.05
            ctrl[3] = 0.90 + k_delta
            ctrl[4] = -0.30 + ap_delta
            ctrl[6] = -0.50 + hp_delta
            ctrl[7] = -0.05
            ctrl[9] = 0.90 + k_delta
            ctrl[10] = -0.30 + ap_delta
        elif stage == "RECOVERY_EXTEND":
            # Extend: push upwards toward upright stand
            ctrl[0] = -0.20 + hp_delta
            ctrl[3] = 0.40 + k_delta
            ctrl[4] = -0.18 + ap_delta
            ctrl[6] = -0.20 + hp_delta
            ctrl[9] = 0.40 + k_delta
            ctrl[10] = -0.18 + ap_delta
        elif stage == "RECOVERY_BALANCE":
            # Balance: active upright posture
            stand = self._default_stand_ctrl()
            ctrl[:] = stand
            ctrl[0] += hp_delta
            ctrl[3] += k_delta
            ctrl[4] += ap_delta
            ctrl[6] += hp_delta
            ctrl[9] += k_delta
            ctrl[10] += ap_delta
        else:  # RECOVERY_HOLD
            # Safe local hold posture
            stand = self._default_stand_ctrl()
            ctrl[:] = stand
            ctrl[0] += hp_delta
            ctrl[3] += k_delta
            ctrl[4] += ap_delta
            ctrl[6] += hp_delta
            ctrl[9] += k_delta
            ctrl[10] += ap_delta

        self.joint_targets = ctrl

    def _step_pure_actuator_feedback(self) -> None:
        """
        500 Hz joint-position PD feedback, converted to bounded actuator torques:
        - NO `xfrc_applied` external forces!
        - NO `qvel` ball velocity modification!
        - Hierarchical and recovery balance policies receive ankle target corrections from IMU.
        - Detects whether the ball has physically been struck by checking if its X velocity
          exceeded 0.25 m/s due to rigid-body foot-to-ball collision.
        """
        decision = self.current_decision
        is_raw_vla = decision is not None and decision.raw_vla_joint_mode

        # Ensure zero external forces on all bodies
        self.data.xfrc_applied[:] = 0.0

        if self.recovery_mode and self.current_recovery_decision is not None:
            if self.current_recovery_decision.recovery_stage == "RECOVERY_BALANCE":
                pelvis_quat = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]
                pelvis_angvel = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]
                roll, pitch, _ = _quat_to_roll_pitch_yaw(pelvis_quat)
                ankle_pitch_corr = -0.35 * pitch - 0.05 * pelvis_angvel[1]
                ankle_roll_corr = -0.35 * roll - 0.05 * pelvis_angvel[0]
                self.joint_targets[4] = np.clip(-0.15 + ankle_pitch_corr, -0.60, 0.45)
                self.joint_targets[10] = np.clip(-0.15 + ankle_pitch_corr, -0.60, 0.45)
                self.joint_targets[5] = np.clip(ankle_roll_corr, -0.20, 0.20)
                self.joint_targets[11] = np.clip(ankle_roll_corr, -0.20, 0.20)
        elif not is_raw_vla and self.current_decision is not None:
            pelvis_quat = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]
            pelvis_angvel = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]
            roll, pitch, _ = _quat_to_roll_pitch_yaw(pelvis_quat)

            ankle_pitch_corr = -0.35 * pitch - 0.05 * pelvis_angvel[1]
            ankle_roll_corr = -0.35 * (roll - 0.06) - 0.05 * pelvis_angvel[0]
            self.joint_targets[4] = np.clip(-0.18 + ankle_pitch_corr, -0.80, 0.45)
            self.joint_targets[5] = np.clip(-0.24 + ankle_roll_corr, -0.26, 0.26)

        joint_angles = self.data.qpos[self._actuator_qpos_adr]
        joint_velocities = self.data.qvel[self._actuator_qvel_adr]
        torques = self.joint_kp * (self.joint_targets - joint_angles)
        torques -= self.joint_kd * joint_velocities
        self.data.ctrl[:] = np.clip(
            torques,
            -self._actuator_torque_limits,
            self._actuator_torque_limits,
        )

        # Record when rigid-body collision actually accelerates the ball forward
        ball_vx = float(self.data.qvel[self.ball_qvel_adr])
        if ball_vx > 0.25:
            self._kick_impulse_applied = True

    def step_physics(self, duration_sec: float) -> None:
        """Advance pure MuJoCo rigid-body physics for `duration_sec` seconds (dt = 0.002s = 500 Hz)."""
        n_steps = max(1, int(round(duration_sec / self.model.opt.timestep)))
        for _ in range(n_steps):
            self._step_pure_actuator_feedback()
            mujoco.mj_step(self.model, self.data)

            bx, by, bz = self.data.xpos[self.ball_body_id]
            if bx >= GOAL_LINE_X and abs(by) <= GOAL_HALF_WIDTH_Y and bz <= GOAL_HEIGHT_Z:
                self.has_scored_goal = True

            pz = self.data.xpos[self.pelvis_body_id][2]
            pelvis_quat = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]
            roll_rad, pitch_rad, _ = _quat_to_roll_pitch_yaw(pelvis_quat)
            roll_deg = float(np.rad2deg(roll_rad))
            pitch_deg = float(np.rad2deg(pitch_rad))
            pelvis_angvel = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]

            if pz < 0.52 or abs(roll_deg) > 42.0 or abs(pitch_deg) > 42.0:
                self.has_fallen = True
                self.recovery_mode = True
                self.kicking_suppressed = True
            elif is_stable_upright(pz, roll_deg, pitch_deg, float(pelvis_angvel[0]), float(pelvis_angvel[1])):
                self.has_fallen = False
                if self.current_recovery_decision is None:
                    self.recovery_mode = False
                    self.kicking_suppressed = False

    def render_camera(self, camera_name: str = "broadcast_cam") -> np.ndarray:
        """Render RGB image (H, W, 3) uint8 from the specified camera."""
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()
