"""Opt-in full-task arm hold. No hardware/Redis dependencies.

The epoch is a host-issued release ticket, NOT a motion command. Only a new
bridge goal carrying the current ticket can release a hold. Cached goals,
restarts and missing heartbeats never release it. Legs/waist stay on TWIST2.
"""
import json
import uuid

import numpy as np

COMMAND_KEY = "bridge:arm_control:v1"
STATUS_KEY = "bridge:arm_hold_status:v1"


class ArmPause:
    def __init__(self):
        self.epoch = uuid.uuid4().hex
        self.held = True
        self.q = None
        self.ref = None
        self.request = None
        self.reason = "startup"
        self.b_down = False
        self.release_ms = None

    def hold(self, measured, reason, *, renew=False):
        if not self.held or self.q is None or renew:
            self.epoch = uuid.uuid4().hex
            # Repeated pause packets must not make the hold target creep.
            if not self.held or self.q is None:
                measured = np.asarray(measured, dtype=float)
                if measured.shape != (29,) or not np.isfinite(measured).all():
                    raise ValueError("arm hold requires finite measured q29")
                self.q = measured[15:29].copy()
            self.release_ms = None
        self.held = True
        self.reason = reason

    def update(self, raw, measured, b_pressed, now_ms):
        if self.q is None:
            self.hold(measured, "startup")
        if b_pressed and not self.b_down:
            # B always stops at the measured pose, including from a ready hold.
            self.q = None
            self.hold(measured, "remote_B", renew=True)
        self.b_down = bool(b_pressed)
        try:
            packet = json.loads(raw) if raw else {}
            fresh = 0 <= now_ms - float(packet["t_ms"]) <= 500
            if not fresh:
                raise ValueError("stale")
            if packet.get("mode") == "hold":
                req = packet.get("request")
                self.hold(measured, "bridge_pause", renew=req != self.request)
                self.request = req
            elif (not b_pressed and packet.get("mode") == "run"
                  and packet.get("epoch") == self.epoch):
                ref = np.asarray(packet["ref"], dtype=float)
                if ref.shape != (35,) or not np.isfinite(ref).all():
                    raise ValueError("bad reference")
                if self.held:
                    self.release_ms = now_ms
                self.held = False
                self.ref = ref.copy()
                self.reason = "fresh_goal"
            elif not self.held:
                self.hold(measured, "invalid_release_ticket")
        except (ValueError, TypeError, KeyError, OverflowError):
            self.hold(measured, "command_missing_or_stale")

    def reference(self, fallback):
        ref = np.asarray(self.ref if self.ref is not None else fallback, dtype=float).copy()
        if self.held:
            ref[21:35] = self.q
        return ref

    def target(self, requested, now_ms):
        out = np.asarray(requested, dtype=np.float32).copy()
        if self.held:
            out[15:29] = self.q
        elif self.release_ms is not None:
            # Bumpless release; never blend towards the pre-pause reference.
            a = np.clip((now_ms - self.release_ms) / 250., 0., 1.)
            out[15:29] = (1 - a) * self.q + a * out[15:29]
        return out

    def status(self, sent, now_ms):
        """Called only AFTER successful DDS write, not on receipt of request."""
        return dict(schema="arm_hold.v1", epoch=self.epoch,
                    mode="holding" if self.held else "running", request=self.request,
                    reason=self.reason, t_ms=now_ms,
                    arm_target=np.asarray(sent)[15:29].tolist())
