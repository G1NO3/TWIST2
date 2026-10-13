"""Offline only: no SDK init, no robot output."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy_real'))
import pico_gmr_intervention_source as source


@pytest.mark.parametrize('side', ['left', 'right'])
def test_pelvis_relative_pose_invariant_to_world_root(side):
    p = np.array([.3, .2, .1])
    r = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    for angle in (0, .7, -1.5, 3.1):
        root_r = Rotation.from_euler('xyz', [.1, -.2, angle]).as_matrix()
        root_p = np.array([5., -3., 2.])
        bodies = {'pelvis': SimpleNamespace(xpos=root_p, xmat=root_r.ravel()),
            f'{side}_wrist_yaw_link': SimpleNamespace(xpos=root_p+root_r@p, xmat=(root_r@r).ravel())}
        pose = source.pelvis_relative_pose(SimpleNamespace(body=bodies.__getitem__), side)
        np.testing.assert_allclose(pose[:3], p, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), r, atol=1e-12)


def test_frozen_body_is_not_refreshed_by_buttons():
    f = source.FreshBody()
    clock = (0,) + (100,) * 24
    assert f.accept(100, clock)
    assert not f.accept(101, clock)
    assert f.accept(102, (0,) + (101,) * 24)
    with pytest.raises(ValueError, match='regressed'):
        f.accept(103, clock)


def test_buttons_read_without_native_controller_poses():
    class SDK:
        def get_time_stamp_ns(self): return 100
        def is_body_data_available(self): return True
        def get_body_joints_pose(self): return np.tile([0, 0, 0, 0, 0, 0, 1], (24, 1))
        def get_body_timestamp_ns(self): return 0
        def get_body_joints_timestamp(self): return [100]*24
        def __getattr__(self, key):
            if key.endswith('_controller_pose'):
                raise AssertionError('Native poses must not gate GMR input')
            if key.endswith('_axis'): return lambda: [0., 0.]
            if key.endswith(('_grip', '_trigger')): return lambda: .2
            if key.endswith(('_button', '_axis_click')): return lambda: False
            raise AttributeError(key)
    sdk = SDK()
    stamp, body, clock, hands = source.read_body_and_buttons(sdk)
    assert stamp == 100 and body.shape == (24, 7)
    assert not hands['left']['key_one']
    times = iter([100, 101])
    sdk.get_time_stamp_ns = lambda: next(times)
    assert source.read_body_and_buttons(sdk) is None


def test_real_mujoco_root_removal():
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting
    r = GeneralMotionRetargeting('xrobot', 'unitree_g1', actual_human_height=1.6, verbose=False)
    data = r.configuration.data
    mujoco.mj_forward(r.model, data)
    before = {s: source.pelvis_relative_pose(data, s) for s in ('left', 'right')}
    data.qpos[:3] += [2, -3, .5]
    xyzw = Rotation.from_euler('xyz', [.2, -.1, 1.4]).as_quat()
    data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(r.model, data)
    for s in before:
        after = source.pelvis_relative_pose(data, s)
        np.testing.assert_allclose(after[:3], before[s][:3], atol=1e-10)
        np.testing.assert_allclose(Rotation.from_quat(after[3:]).as_matrix(),
            Rotation.from_quat(before[s][3:]).as_matrix(), atol=1e-10)


def test_no_robot_transport_imports():
    tree = ast.parse(Path(source.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): modules.extend(n.name for n in node.names)
        if isinstance(node, ast.ImportFrom): modules.append(node.module or '')
    assert not any(any(s in m for s in ('redis', 'unitree_sdk', 'robot_control', 'cyclonedds')) for m in modules)
