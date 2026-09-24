"""Final joint-position/velocity guard before a real G1 DDS write."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class JointSafetyResult:
    target: np.ndarray
    valid: bool
    position_clipped: int
    velocity_clipped: int
    max_requested_step_rad: float
    max_applied_step_rad: float
    reason: str


class JointCommandSafety:
    """Clip a 29-D command by URDF limits and target slew rate.

    URDF velocity numbers are hardware maxima, not suitable target slew
    rates.  ``velocity_scale`` applies a conservative fraction (default 10%)
    per 20 ms cycle.  The first call starts from measured joints, never from
    an arbitrary default pose.
    """

    def __init__(self, urdf_path, control_dt: float,
                 velocity_scale: float = 0.10,
                 position_margin_rad: float = 0.02):
        if control_dt <= 0 or not 0 < velocity_scale <= 1:
            raise ValueError("control_dt and velocity_scale must be positive")
        nodes = {node.attrib["name"]: node
                 for node in ET.parse(urdf_path).getroot().findall("joint")}
        lower, upper, velocity = [], [], []
        for name in JOINT_NAMES:
            node = nodes.get(name)
            limit = None if node is None else node.find("limit")
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError(f"URDF has no finite limits for {name}")
            lower.append(float(limit.get("lower")) + position_margin_rad)
            upper.append(float(limit.get("upper")) - position_margin_rad)
            velocity.append(float(limit.get("velocity")))
        self.lower = np.asarray(lower, np.float64)
        self.upper = np.asarray(upper, np.float64)
        if np.any(self.lower >= self.upper):
            raise ValueError("position margin consumes a joint range")
        self.max_step = np.asarray(velocity, np.float64) * float(control_dt) * float(velocity_scale)
        self.previous: np.ndarray | None = None

    def reset(self, measured_q) -> None:
        measured = np.asarray(measured_q, np.float64).reshape(-1)
        if measured.shape != (29,) or not np.all(np.isfinite(measured)):
            raise ValueError("measured_q must contain 29 finite values")
        self.previous = np.clip(measured, self.lower, self.upper)

    def filter(self, requested, measured_q=None) -> JointSafetyResult:
        raw = np.asarray(requested, np.float64).reshape(-1)
        if self.previous is None:
            if measured_q is None:
                raise RuntimeError("first safety filter call requires measured_q")
            self.reset(measured_q)
        assert self.previous is not None
        if raw.shape != (29,) or not np.all(np.isfinite(raw)):
            return JointSafetyResult(
                self.previous.astype(np.float32).copy(), False, 0, 0,
                0.0, 0.0, "invalid_command_hold")
        positioned = np.clip(raw, self.lower, self.upper)
        position_count = int(np.count_nonzero(np.abs(positioned - raw) > 1e-12))
        requested_step = positioned - self.previous
        applied_step = np.clip(requested_step, -self.max_step, self.max_step)
        velocity_count = int(np.count_nonzero(np.abs(applied_step - requested_step) > 1e-12))
        safe = np.clip(self.previous + applied_step, self.lower, self.upper)
        self.previous = safe
        reason = "position_and_velocity_clipped" if position_count and velocity_count \
            else "position_clipped" if position_count \
            else "velocity_clipped" if velocity_count else "ok"
        return JointSafetyResult(
            safe.astype(np.float32), True, position_count, velocity_count,
            float(np.max(np.abs(requested_step))),
            float(np.max(np.abs(applied_step))), reason)
