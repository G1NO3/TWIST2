"""Fail-closed behavior for an Inspire hand holding a tool."""

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


DEPLOY_REAL = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_REAL))

from robot_control.inspire_hand_hil import InspireHandController  # noqa: E402


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
    source = (DEPLOY_REAL / "server_low_level_g1_hil.py").read_text()
    assert "open_on_close=not hil_safety" in source
    assert "self.hand_ctrl.current_hold_targets()" in source


def _command_controller():
    hand = _bare_controller()
    hand._cmd_lock = threading.Lock()
    hand.enable_left = hand.enable_right = True
    hand._target_left = np.full(6, 1000, dtype=np.int32)
    hand._target_right = np.array([0, 0, 0, 0, 639, 777], dtype=np.int32)
    hand._target_dirty_left = hand._target_dirty_right = False
    hand._last_reset_token = None
    return hand


def test_each_reset_resends_identical_grasp_once_without_opening():
    hand = _command_controller()
    left, right = hand._target_left.copy(), hand._target_right.copy()
    hand.ctrl_dual_hand(left, right)
    assert not hand._target_dirty_left and not hand._target_dirty_right
    for token in (b"first-Y", b"second-Y"):
        hand.ctrl_dual_hand(left, right, reset_token=token)
        assert hand._target_dirty_left and hand._target_dirty_right
        np.testing.assert_array_equal(hand._target_right, right)
        hand._target_dirty_left = hand._target_dirty_right = False
        hand.ctrl_dual_hand(left, right, reset_token=token)
        assert not hand._target_dirty_left and not hand._target_dirty_right


@pytest.mark.parametrize("failure", ["exception", "modbus_error"])
def test_failed_reset_write_retries_latest_target(failure):
    hand = _command_controller()
    hand._stop_event = threading.Event()
    hand._worker_period_s = 0.001
    hand._dev_kwargs = {}
    hand._note_ok = lambda *_: None
    hand._note_error = lambda *_: None
    hand._read_registers_signed = lambda *_: [0] * 6
    hand._read_registers_bytes = lambda *_: [0] * 6
    hand._read_tactile = lambda *_: None
    hand.Rtactile = np.zeros(1062, dtype=np.uint16)
    hand.ctrl_dual_hand(hand._target_left, hand._target_right, reset_token=b"Y")
    writes = []

    def write(address, values, **kwargs):
        writes.append((address, values))
        if len(writes) == 1:
            # A newer target arrives while the previous write is in flight.
            hand.ctrl_dual_hand(hand._target_left, [0, 0, 0, 0, 640, 778])
            if failure == "exception":
                raise IOError("connection interrupted")
            return SimpleNamespace(isError=lambda: True)
        hand._stop_event.set()
        return SimpleNamespace(isError=lambda: False)

    worker = threading.Thread(target=hand._worker_loop,
                              args=("right", SimpleNamespace(write_registers=write)))
    worker.start()
    worker.join(timeout=2)
    hand._stop_event.set()
    worker.join(timeout=1)
    assert writes == [(1486, [0, 0, 0, 0, 639, 777]),
                      (1486, [0, 0, 0, 0, 640, 778])]
