"""Fail-closed behavior for an Inspire hand holding a tool."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


DEPLOY_REAL = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_REAL))

from robot_control.inspire_hand_wrapper import InspireHandController  # noqa: E402


def _bare_controller(left=None, right=None):
    hand = InspireHandController.__new__(InspireHandController)
    hand._state_lock = threading.Lock()
    hand.left_hand_state_array = np.asarray(
        [1000, 900, 800, 700, 600, 500] if left is None else left,
        dtype=np.float32)
    hand.right_hand_state_array = np.asarray(
        [0, 0, 0, 0, 670, 653] if right is None else right,
        dtype=np.float32)
    return hand


def test_current_hold_targets_are_validated_copies():
    hand = _bare_controller()
    left, right = hand.current_hold_targets()
    np.testing.assert_array_equal(left, [1000, 900, 800, 700, 600, 500])
    np.testing.assert_array_equal(right, [0, 0, 0, 0, 670, 653])

    left[0] = 0
    assert hand.left_hand_state_array[0] == 1000

    hand.right_hand_state_array[2] = np.nan
    with pytest.raises(RuntimeError, match="non-finite right"):
        hand.current_hold_targets()
    hand.right_hand_state_array[2] = 1001
    with pytest.raises(RuntimeError, match="out-of-range right"):
        hand.current_hold_targets()


def test_modbus_read_failure_never_becomes_fake_zero_feedback():
    hand = InspireHandController.__new__(InspireHandController)
    hand._read_holding = lambda *_args: SimpleNamespace(
        isError=lambda: True)
    with pytest.raises(IOError, match="Modbus error"):
        hand._read_registers_signed(object(), 1534, 6)
    with pytest.raises(IOError, match="Modbus error"):
        hand._read_registers_bytes(object(), 1618, 3)

    hand._read_holding = lambda *_args: SimpleNamespace(
        isError=lambda: False, registers=[1, 2])
    with pytest.raises(IOError, match="short Modbus read"):
        hand._read_registers_signed(object(), 1534, 6)


class _Client:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_hil_close_preserves_last_hand_setpoints():
    hand = InspireHandController.__new__(InspireHandController)
    hand._stop_event = threading.Event()
    hand._worker_left = None
    hand._worker_right = None
    hand.open_on_close = False
    hand.left_client = _Client()
    hand.right_client = _Client()
    writes = []
    hand._bootstrap_write_default_sync = lambda: writes.append("open")

    hand.close()

    assert not writes
    assert hand.left_client.closed and hand.right_client.closed


def test_real_hil_host_selects_hold_on_close_and_measured_takeover():
    source = (DEPLOY_REAL / "server_low_level_g1_real.py").read_text()
    assert "open_on_close=not hil_safety" in source
    assert "self.hand_ctrl.current_hold_targets()" in source
