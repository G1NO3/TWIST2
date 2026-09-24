"""Validation and fail-safe shaping for HIL gravity feedforward."""

from __future__ import annotations

import json

import numpy as np


# Independent host-side limits for the seven right-arm motors.  These mirror
# the correction sidecar's shoulder/elbow and wrist clamps, but are repeated
# here deliberately: Redis is an untrusted process boundary and the DDS host
# must never rely on the writer to enforce actuator-facing limits.
RIGHT_ARM_TAU_LIMITS_NM = np.asarray(
    [14.0, 14.0, 14.0, 14.0, 3.0, 3.0, 3.0], dtype=np.float32)


def decode_right_arm_gravity(raw, *, now_ms: float, stale_ms: float):
    """Return a validated, clipped 7-D right-arm torque sample.

    The wire payload contains both arms in joint order (left then right):
    ``{"t_ms": ..., "tau": [14 floats]}``.  Invalid, stale, or future-dated
    data returns ``(None, reason)`` so the caller can fade the last valid
    sample out instead of applying an abrupt zero-torque step.
    """
    if raw is None:
        return None, "missing"
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "malformed_json"
    if not isinstance(payload, dict):
        return None, "malformed_payload"

    tau_raw = payload.get("tau")
    if not isinstance(tau_raw, list) or len(tau_raw) != 14:
        return None, "bad_tau_shape"
    try:
        stamp = float(payload.get("t_ms"))
        tau = np.asarray(tau_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None, "non_numeric"
    if tau.shape != (14,):
        return None, "bad_tau_shape"
    if not np.isfinite(stamp) or not np.all(np.isfinite(tau)):
        return None, "non_finite"

    age_ms = float(now_ms) - stamp
    if age_ms < 0.0:
        return None, "future_timestamp"
    if age_ms > float(stale_ms):
        return None, "stale"

    right = np.clip(tau[7:14], -RIGHT_ARM_TAU_LIMITS_NM,
                    RIGHT_ARM_TAU_LIMITS_NM)
    return right.astype(np.float32), "ok"
