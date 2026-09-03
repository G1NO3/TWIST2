"""Wire-contract tests for the physical Unitree HIL remote."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


DEPLOY_REAL = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_REAL))

from robot_control.hil_remote import key_pressed, remote_snapshot  # noqa: E402


def test_key_pressed_supports_button_chords():
    keys = 0x0800 | 0x0100
    assert key_pressed(keys, 0x0800)
    assert key_pressed(keys, 0x0100)
    assert not key_pressed(keys, 0x0400)


def test_remote_snapshot_is_json_safe_and_clamps_axes():
    remote = SimpleNamespace(
        keys=0x10800, lx=1.4, ly=-0.25, rx=0.5, ry=-1.2)
    got = remote_snapshot(remote, 1234)
    assert got == {
        "source": "unitree_remote",
        "keys": 0x0800,
        "lx": 1.0,
        "ly": -0.25,
        "rx": 0.5,
        "ry": -1.0,
        "timestamp_ms": 1234,
    }


def test_remote_snapshot_rejects_nonfinite_axis():
    remote = SimpleNamespace(keys=0, lx=0.0, ly=math.nan, rx=0.0, ry=0.0)
    with pytest.raises(ValueError, match="ly"):
        remote_snapshot(remote, 1234)
