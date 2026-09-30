"""No robot I/O: verify the actual PD law, mode reversals and final safety."""
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy_real.robot_control.arm_handover import ArmCommand, ArmHandover, blend_commands
from deploy_real.robot_control.joint_command_safety import JointCommandSafety


def command(q=.1, kp=1., kd=1., tau=0.):
    return ArmCommand(*(np.full(7, v, dtype=float) for v in (q, kp, kd, tau)))


def torque(c, q, dq):
    return 40*c.kp*(c.q-q) - 2*c.kd*dq + c.tau


def test_blend_preserves_support_with_different_gains():
    tracker = command(q=.1)  # 40*.1 = 4 Nm support
    pd = command(q=0., kp=3., kd=1.7, tau=4.)
    # Previous angle interpolation creates a 50% force overshoot halfway.
    old = command(q=.05, kp=2., kd=1.35, tau=2.)
    np.testing.assert_allclose(torque(old, 0., 0.), 6.)
    for a in np.linspace(0, 1, 31):
        mixed = blend_commands(tracker, pd, a)
        np.testing.assert_allclose(torque(mixed, 0., 0.), 4., atol=1e-12)
        for q, dq in [(.04, .3), (-.05, -.2)]:
            np.testing.assert_allclose(torque(mixed, q, dq),
                                       (1-a)*torque(tracker, q, dq)+a*torque(pd, q, dq))


def test_both_directions_keep_first_command_then_reach_endpoint():
    handover = ArmHandover(.02, .25)
    start = command()
    handover.record_sent(start)
    for direct, incoming in [(True, command(q=0., kp=3., kd=1.7, tau=4.)),
                             (False, command())]:
        first = handover.update(incoming, direct)
        for field in ('q', 'kp', 'kd', 'tau'):
            np.testing.assert_array_equal(getattr(first, field), getattr(start, field))
        assert handover.event == 'start'
        handover.record_sent(first)
        for _ in range(13):
            out = handover.update(incoming, direct)
            np.testing.assert_allclose(torque(out, 0., 0.), 4., atol=1e-12)
            handover.record_sent(out)
        assert handover.event == 'complete'
        for field in ('q', 'kp', 'kd', 'tau'):
            np.testing.assert_array_equal(getattr(out, field), getattr(incoming, field))
        start = out


def test_reversal_uses_last_post_safety_command_not_original_origin():
    handover = ArmHandover(.02, .25)
    handover.record_sent(command())
    incoming = command(q=-.3, kp=3., tau=4.)
    for _ in range(5):
        out = handover.update(incoming, True)
        # Emulate a post-safety clipped target.
        out.q = np.clip(out.q, .08, .1)
        handover.record_sent(out)
    actual = out.copy()
    released = handover.update(command(q=.5), False)
    np.testing.assert_array_equal(torque(released, .04, .3), torque(actual, .04, .3))


def test_tracker_only_commands_are_unchanged():
    handover = ArmHandover(.02, .25)
    for i in range(10):
        incoming = command(q=i*.01)
        out = handover.update(incoming, False)
        assert handover.event is None
        np.testing.assert_array_equal(out.q, incoming.q)
        handover.record_sent(out)


def test_motor_path_records_post_safety_target_and_preserves_other_joints():
    # Compile only the production motor-send method, without importing DDS,
    # cameras, policy weights or the robot SDK.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root/'deploy_real/server_low_level_g1_hil.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == '_send_robot_action')
    namespace = dict(np=np, json=json, time=time, ArmCommand=ArmCommand)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<motor-send>', 'exec'), namespace)
    writes = []
    env = SimpleNamespace(send_robot_action=lambda q, *a, **kw: writes.append((q.copy(), kw)))
    pipe = SimpleNamespace(set=lambda *a: None, execute=lambda: None)
    safety = JointCommandSafety(str(root/'assets/g1/g1_29dof_rev_1_0.urdf'), .02)
    q = ((safety.lower+safety.upper)/2).astype(np.float32)
    obj = SimpleNamespace(arm_override_enabled=True, arm_handover=ArmHandover(.02, .25),
                          command_safety=safety, _last_measured_q=q.copy(),
                          _last_measured_dq=np.zeros(29), _arm_override_reason='test',
                          env=env, redis_pipeline=pipe)
    send = lambda *a, **kw: namespace['_send_robot_action'](obj, *a, **kw)
    send(q)
    start = writes[-1]
    pd = q.copy(); pd[22:29] += .3
    kp, kd, tau = np.ones(29), np.ones(29), np.zeros(29)
    kp[22:], kd[22:], tau[22:] = 3., 1.7, 4.
    send(pd, kp_motor_scale=kp, kd_motor_scale=kd, tau_ff=tau, arm_override_active=True)
    np.testing.assert_array_equal(writes[-1][0], start[0])
    for field in ('kp_motor_scale', 'kd_motor_scale', 'tau_ff'):
        np.testing.assert_array_equal(writes[-1][1][field], start[1][field])
    for _ in range(13):
        before = writes[-1][0]
        send(pd, kp_motor_scale=kp, kd_motor_scale=kd, tau_ff=tau, arm_override_active=True)
        after, values = writes[-1]
        assert np.all(np.abs(after-before) <= safety.max_step+1e-6)
        np.testing.assert_array_equal(after[:22], q[:22])
        np.testing.assert_array_equal(values['kp_motor_scale'][:22], np.ones(22))
        np.testing.assert_array_equal(obj.arm_handover.last.q, after[22:])
    # The stale-reference path also enters through this sole send method.
    previous = writes[-1]
    send(previous[0])
    for field in ('kp_motor_scale', 'kd_motor_scale', 'tau_ff'):
        np.testing.assert_array_equal(writes[-1][1][field], previous[1][field])
    # A malformed policy output must keep the complete safe servo command,
    # rather than throwing before the existing invalid-command hold guard.
    previous = writes[-1]
    bad = pd.copy(); bad[24] = np.nan
    send(bad)
    np.testing.assert_array_equal(writes[-1][0], previous[0])
    for field in ('kp_motor_scale', 'kd_motor_scale', 'tau_ff'):
        np.testing.assert_array_equal(writes[-1][1][field], previous[1][field])


@pytest.mark.parametrize('dt,duration', [(0,.25), (.02,0), (.02,float('nan'))])
def test_invalid_timing_rejected(dt, duration):
    with pytest.raises(ValueError):
        ArmHandover(dt, duration)
