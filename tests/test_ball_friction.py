"""Measure ball stopping distance using the actual scene and contact solver."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soccer_pk_env import G1_SOCCER_XML_PATH


def stopping_distance(sliding_friction=None):
    model = mujoco.MjModel.from_xml_path(str(G1_SOCCER_XML_PATH))
    pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
    if sliding_friction is not None:
        model.pair_friction[pair, :2] = sliding_friction
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    qpos = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    data.qpos[qpos:qpos + 3] = [-8.0, -8.0, 0.103]
    data.qvel[dof] = 4.25
    still_since = None
    for _ in range(int(30 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if np.linalg.norm(data.qvel[dof:dof + 3]) < 0.08:
            if still_since is None:
                still_since = data.time
            elif data.time - still_since > 0.5:
                return float(data.qpos[qpos] + 8.0), float(data.time)
        else:
            still_since = None
    raise AssertionError("Ball did not stop within 30 simulation seconds.")


class TestBallFriction(unittest.TestCase):
    def test_stopping_distance_is_about_one_third(self):
        baseline, baseline_time = stopping_distance(0.04)
        distance, stop_time = stopping_distance()
        ratio = distance / baseline
        self.assertGreater(ratio, 0.30)
        self.assertLess(ratio, 0.37)
        self.assertLess(stop_time, baseline_time)

    def test_only_ball_floor_pair_changes_friction(self):
        model = mujoco.MjModel.from_xml_path(str(G1_SOCCER_XML_PATH))
        pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
        ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.assertEqual({model.pair_geom1[pair], model.pair_geom2[pair]}, {ball, floor})
        np.testing.assert_allclose(model.geom_friction[[ball, floor], 0], 0.04)
        np.testing.assert_allclose(model.pair_friction[pair, :2], 0.16)
        self.assertEqual(model.pair_dim[pair], 3)


if __name__ == "__main__":
    unittest.main()
