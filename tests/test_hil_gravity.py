import json

import numpy as np

from deploy_real.robot_control.hil_gravity import (
    RIGHT_ARM_TAU_LIMITS_NM,
    decode_right_arm_gravity,
)


def _payload(tau=None, stamp=1000):
    return json.dumps({
        "t_ms": stamp,
        "tau": list(range(14)) if tau is None else tau,
    })


def test_valid_sample_selects_right_arm_and_clips_at_host():
    tau = [0.0] * 7 + [100.0, -100.0, 8.0, -8.0, 9.0, -9.0, 1.5]
    result, reason = decode_right_arm_gravity(
        _payload(tau), now_ms=1100, stale_ms=250)

    assert reason == "ok"
    np.testing.assert_allclose(
        result, [14.0, -14.0, 8.0, -8.0, 3.0, -3.0, 1.5])
    assert np.all(np.abs(result) <= RIGHT_ARM_TAU_LIMITS_NM)


def test_rejects_stale_and_future_timestamps():
    assert decode_right_arm_gravity(
        _payload(stamp=700), now_ms=1000, stale_ms=250)[1] == "stale"
    assert decode_right_arm_gravity(
        _payload(stamp=1001), now_ms=1000, stale_ms=250)[1] \
        == "future_timestamp"


def test_rejects_malformed_shape_and_non_finite_torque():
    assert decode_right_arm_gravity(
        "[]", now_ms=1000, stale_ms=250)[1] == "malformed_payload"
    assert decode_right_arm_gravity(
        _payload([0.0] * 13), now_ms=1000, stale_ms=250)[1] \
        == "bad_tau_shape"
    assert decode_right_arm_gravity(
        _payload([[0.0]] * 14), now_ms=1000, stale_ms=250)[1] \
        == "bad_tau_shape"
    bad = [0.0] * 14
    bad[9] = float("nan")
    assert decode_right_arm_gravity(
        _payload(bad), now_ms=1000, stale_ms=250)[1] == "non_finite"
