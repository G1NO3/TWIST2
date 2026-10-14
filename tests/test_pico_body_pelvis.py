"""Offline only: body_pelvis wrist frames (no SDK init, no robot output)."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy_real'))
import pico_gmr_intervention_source as source

OFF = source.gmr_offsets()


def human(pelvis_yaw=0., pelvis_tilt=0., pelvis_p=(1., 2., .9), wrists=None):
    """World (right-handed, z up) human joints whose G1-convention frames are given."""
    g1_pelvis = Rotation.from_euler('ZY', [pelvis_yaw, pelvis_tilt])  # yaw, then lean about own y
    pelvis_q = (g1_pelvis * OFF['Pelvis'].inv()).as_quat(scalar_first=True)
    out = {'Pelvis': [list(pelvis_p), pelvis_q.tolist()]}
    H = Rotation.from_euler('z', pelvis_yaw)
    for joint, (p, r) in (wrists or {}).items():
        world_p = np.asarray(pelvis_p) + H.apply(p)
        world_r = H * r * OFF[joint].inv()
        out[joint] = [world_p.tolist(), world_r.as_quat(scalar_first=True).tolist()]
    return out


def poses(h):
    return source.body_relative_wrists(h, OFF)[0]


def test_gmr_offsets_come_from_gmr_config():
    assert set(OFF) == {'Pelvis', 'Left_Wrist', 'Right_Wrist'}
    np.testing.assert_allclose(OFF['Pelvis'].as_quat(scalar_first=True), [-.5, .5, -.5, -.5])


@pytest.mark.parametrize('yaw', [0., .8, -2., 3.1])
@pytest.mark.parametrize('tilt', [0., .35])
def test_wrist_offsets_are_in_the_pelvis_heading_frame(yaw, tilt):
    r = Rotation.from_rotvec([.3, -.2, .5])
    w = {'Left_Wrist': ([.3, .2, .1], r), 'Right_Wrist': ([.25, -.2, .05], r.inv())}
    got = poses(human(yaw, tilt, wrists=w))
    for side, joint in (('left', 'Left_Wrist'), ('right', 'Right_Wrist')):
        np.testing.assert_allclose(got[side][:3], w[joint][0], atol=1e-9)
        np.testing.assert_allclose(Rotation.from_quat(got[side][3:]).as_matrix(), w[joint][1].as_matrix(), atol=1e-9)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_body_relative_motion_maps_to_forward_left_up(axis):
    start = {'Right_Wrist': ([.3, -.2, .1], Rotation.identity()), 'Left_Wrist': ([.3, .2, .1], Rotation.identity())}
    moved = dict(start)
    p = np.array(start['Right_Wrist'][0]); p[axis] += .2
    moved['Right_Wrist'] = (p.tolist(), Rotation.identity())
    for yaw in (0., 1.2, -2.5):   # operator heading never changes the mapping
        d = np.subtract(poses(human(yaw, wrists=moved))['right'][:3], poses(human(yaw, wrists=start))['right'][:3])
        np.testing.assert_allclose(d, np.eye(3)[axis]*.2, atol=1e-9)


def test_pure_wrist_rotation_does_not_translate():
    p = [.3, .2, .1]
    a = poses(human(wrists={'Left_Wrist': (p, Rotation.identity()), 'Right_Wrist': (p, Rotation.identity())}))
    b = poses(human(wrists={'Left_Wrist': (p, Rotation.from_rotvec([1.5, 0, 0])), 'Right_Wrist': (p, Rotation.identity())}))
    np.testing.assert_allclose(a['left'][:3], b['left'][:3], atol=1e-12)


def test_vertical_pelvis_heading_is_refused():
    with pytest.raises(ValueError, match='heading'):
        poses(human(pelvis_tilt=np.pi/2*.98, wrists={'Left_Wrist': ([0, 0, 0], Rotation.identity()),
                                                     'Right_Wrist': ([0, 0, 0], Rotation.identity())}))
