"""Continuous PD command transfer between a tracker and a gravity-backed arm.

The motor law is kp * (q_target - q) - kd * dq + tau_ff. Interpolating
q_target and kp separately adds an unwanted cross term. Interpolate
kp*q_target instead so the nominal motor torque is a convex blend at any
measured q/dq. Joint safety still has final authority over q_target.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class ArmCommand:
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray

    def copy(self):
        return ArmCommand(*(np.asarray(v, dtype=np.float64).copy()
                            for v in (self.q, self.kp, self.kd, self.tau)))

    def validate(self):
        if any(np.shape(v) != (7,) or not np.isfinite(v).all()
               for v in (self.q, self.kp, self.kd, self.tau)):
            raise ValueError('Arm commands require seven finite values per field')
        if np.any(self.kp <= 0) or np.any(self.kd < 0):
            raise ValueError('Arm gains must have positive kp and nonnegative kd')


def blend_commands(previous, incoming, weight):
    previous.validate()
    incoming.validate()
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Blend weight must be within [0, 1]')
    if weight == 0:
        return previous.copy()
    if weight == 1:
        return incoming.copy()
    a = float(weight)
    kp = (1-a)*previous.kp + a*incoming.kp
    return ArmCommand(
        ((1-a)*previous.kp*previous.q + a*incoming.kp*incoming.q) / kp,
        kp, (1-a)*previous.kd + a*incoming.kd,
        (1-a)*previous.tau + a*incoming.tau)


class ArmHandover:
    def __init__(self, dt, duration):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Handover timing must be finite and positive')
        self.step = min(1., float(dt)/float(duration))
        self.last = self.origin = None
        self.direct = False
        self.progress = 1.
        self.event = None

    def update(self, incoming, direct):
        incoming.validate()
        self.event = None
        direct = bool(direct)
        if self.last is None:
            # The host starts in its regular controller; no previous command
            # exists until the first post-safety DDS write is acknowledged.
            self.direct = direct
            return incoming.copy()
        if direct != self.direct:
            self.direct = direct
            self.origin = self.last.copy()
            self.progress = 0.
            self.event = 'start'
        elif self.origin is not None:
            self.progress = min(1., self.progress + self.step)
        if self.origin is None:
            return incoming.copy()
        # Smooth start/end slopes, without wall-clock catch-up after a stall.
        weight = self.progress**2 * (3 - 2*self.progress)
        result = blend_commands(self.origin, incoming, weight)
        if self.progress >= 1.:
            self.origin = None
            self.event = 'complete'
        return result

    def record_sent(self, command):
        """Record post-safety targets, including a reversal during transition."""
        command.validate()
        self.last = command.copy()
