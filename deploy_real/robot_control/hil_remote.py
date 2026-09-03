"""Pure helpers for publishing the G1 wireless remote to the HIL actor.

This module deliberately does not import the Unitree SDK, Redis or NumPy so
its wire contract can be tested on a development machine.
"""

from __future__ import annotations

import math


KEY_MASKS = {
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "start": 0x0004,
    "select": 0x0008,
}

REDIS_REMOTE_KEY = "hil:unitree_remote"
REDIS_HEARTBEAT_KEY = "hil:unitree_remote_heartbeat_ms"
SOURCE = "unitree_remote"


def key_pressed(keys: int, mask: int) -> bool:
    """Test a button bit without breaking multi-button chords."""
    return bool(int(keys) & int(mask))


def _axis(remote, name: str) -> float:
    value = float(getattr(remote, name))
    if not math.isfinite(value):
        raise ValueError(f"wireless remote axis {name} is not finite")
    return max(-1.0, min(1.0, value))


def remote_snapshot(remote, timestamp_ms: int) -> dict:
    """Return the source-specific, last-value Redis payload."""
    timestamp_ms = int(timestamp_ms)
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    return {
        "source": SOURCE,
        "keys": int(getattr(remote, "keys")) & 0xFFFF,
        "lx": _axis(remote, "lx"),
        "ly": _axis(remote, "ly"),
        "rx": _axis(remote, "rx"),
        "ry": _axis(remote, "ry"),
        "timestamp_ms": timestamp_ms,
    }
