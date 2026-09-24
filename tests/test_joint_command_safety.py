from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "deploy_real"))

from robot_control.joint_command_safety import JointCommandSafety  # noqa: E402


URDF = ROOT / "assets/g1/g1_29dof_rev_1_0.urdf"


def test_position_and_velocity_are_both_limited():
    guard = JointCommandSafety(URDF, control_dt=0.02,
                               velocity_scale=0.1,
                               position_margin_rad=0.02)
    measured = np.zeros(29)
    guard.reset(measured)
    requested = np.full(29, 100.0)
    got = guard.filter(requested)
    assert got.valid
    assert got.position_clipped == 29
    assert got.velocity_clipped > 0
    assert np.all(got.target <= guard.upper)
    assert np.all(np.abs(got.target) <= guard.max_step + 1e-8)


def test_nonfinite_command_holds_previous():
    guard = JointCommandSafety(URDF, 0.02)
    guard.reset(np.zeros(29))
    first = guard.filter(np.full(29, 0.01)).target
    bad = np.full(29, 0.02)
    bad[3] = np.nan
    got = guard.filter(bad)
    assert not got.valid
    np.testing.assert_array_equal(got.target, first)
