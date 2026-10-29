"""
Unitree G1 Official RL Locomotion Policy (`motion.pt` from `unitree_rl_gym`) executed in pure NumPy.

Architecture of `deploy/pre_train/g1/motion.pt` (`PolicyExporterLSTM`):
- Input observation: 47-dim float32 vector at 50 Hz (`dt = 0.02s`, `control_decimation = 10` of `0.002s`):
  - `obs[0:3]`: base angular velocity `omega * 0.25`
  - `obs[3:6]`: projected gravity orientation in pelvis frame
  - `obs[6:9]`: velocity command `[vx, vy, vyaw] * [2.0, 2.0, 0.25]`
  - `obs[9:21]`: 12 leg joint angle offsets `(q_leg - default_angles) * 1.0`
  - `obs[21:33]`: 12 leg joint velocities `dq_leg * 0.05`
  - `obs[33:45]`: previous 12-dim action `action`
  - `obs[45:47]`: gait clock `[sin(2*pi*phase), cos(2*pi*phase)]` with `period = 0.8s`
- Network layers:
  1. `LSTM(input_size=47, hidden_size=64)` (`data/6`: `w_ih`, `data/7`: `w_hh`, `data/8`: `b_ih`, `data/9`: `b_hh`)
  2. `Linear(64 -> 32)` (`data/2`: weight, `data/3`: bias)
  3. `ELU(alpha=1.0)`
  4. `Linear(32 -> 12)` (`data/4`: weight, `data/5`: bias)
- Output: 12 leg joint target offsets (`target_q = action * 0.25 + default_angles`).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np


DEFAULT_LEG_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32,
)
LEG_KPS = np.array(
    [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32
)
LEG_KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * (np.exp(np.clip(x, -30.0, 0.0)) - 1.0))


def get_gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute projected gravity vector in the robot's pelvis frame from MuJoCo quaternion (w, x, y, z)."""
    qw, qx, qy, qz = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)
    g[1] = -2.0 * (qz * qy + qw * qx)
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return g


class UnitreeG1RLPolicyNumPy:
    """Pure NumPy inference engine for Unitree's official `motion.pt` TorchScript LSTM policy."""

    def __init__(self, pt_path: str | Path) -> None:
        pt_path = Path(pt_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Unitree G1 RL policy not found at: {pt_path}")

        with zipfile.ZipFile(pt_path, "r") as z:
            self.fc0_w = np.frombuffer(
                z.read("policy_lstm_1/data/2"), dtype=np.float32
            ).reshape(32, 64).copy()
            self.fc0_b = np.frombuffer(
                z.read("policy_lstm_1/data/3"), dtype=np.float32
            ).reshape(32).copy()
            self.fc2_w = np.frombuffer(
                z.read("policy_lstm_1/data/4"), dtype=np.float32
            ).reshape(12, 32).copy()
            self.fc2_b = np.frombuffer(
                z.read("policy_lstm_1/data/5"), dtype=np.float32
            ).reshape(12).copy()
            self.w_ih = np.frombuffer(
                z.read("policy_lstm_1/data/6"), dtype=np.float32
            ).reshape(256, 47).copy()
            self.w_hh = np.frombuffer(
                z.read("policy_lstm_1/data/7"), dtype=np.float32
            ).reshape(256, 64).copy()
            self.b_ih = np.frombuffer(
                z.read("policy_lstm_1/data/8"), dtype=np.float32
            ).reshape(256).copy()
            self.b_hh = np.frombuffer(
                z.read("policy_lstm_1/data/9"), dtype=np.float32
            ).reshape(256).copy()

        self.h = np.zeros(64, dtype=np.float32)
        self.c = np.zeros(64, dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_dof_pos = DEFAULT_LEG_ANGLES.copy()

    def reset(self) -> None:
        self.h.fill(0.0)
        self.c.fill(0.0)
        self.last_action.fill(0.0)
        self.target_dof_pos = DEFAULT_LEG_ANGLES.copy()

    def step_50hz(
        self,
        q_leg12: np.ndarray,
        dq_leg12: np.ndarray,
        quat_wxyz: np.ndarray,
        omega_pelvis: np.ndarray,
        cmd_vx_vy_vyaw: Tuple[float, float, float],
        sim_time_sec: float,
    ) -> np.ndarray:
        """Run one 50 Hz RL policy step and return the 12 target leg joint angles."""
        qj = (q_leg12.astype(np.float32) - DEFAULT_LEG_ANGLES) * 1.0
        dqj = dq_leg12.astype(np.float32) * 0.05
        grav = get_gravity_orientation(quat_wxyz)
        omega = omega_pelvis.astype(np.float32) * 0.25
        cmd = np.array(cmd_vx_vy_vyaw, dtype=np.float32) * CMD_SCALE

        period = 0.8
        phase = (sim_time_sec % period) / period
        sin_p = np.float32(np.sin(2.0 * np.pi * phase))
        cos_p = np.float32(np.cos(2.0 * np.pi * phase))

        obs = np.zeros(47, dtype=np.float32)
        obs[0:3] = omega
        obs[3:6] = grav
        obs[6:9] = cmd
        obs[9:21] = qj
        obs[21:33] = dqj
        obs[33:45] = self.last_action
        obs[45:47] = [sin_p, cos_p]

        # LSTM cell forward: gates = W_ih @ x + b_ih + W_hh @ h + b_hh
        gates = self.w_ih @ obs + self.b_ih + self.w_hh @ self.h + self.b_hh
        i_gate = _sigmoid(gates[0:64])
        f_gate = _sigmoid(gates[64:128])
        g_gate = np.tanh(gates[128:192])
        o_gate = _sigmoid(gates[192:256])

        self.c = f_gate * self.c + i_gate * g_gate
        self.h = o_gate * np.tanh(self.c)

        # Actor MLP: Linear(64->32) -> ELU -> Linear(32->12)
        x1 = _elu(self.fc0_w @ self.h + self.fc0_b)
        action = self.fc2_w @ x1 + self.fc2_b

        self.last_action = action.copy()
        self.target_dof_pos = action * 0.25 + DEFAULT_LEG_ANGLES
        return self.target_dof_pos.copy()
