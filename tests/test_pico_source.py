import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pico_source", Path(__file__).resolve().parents[1] / "deploy_real/pico_intervention_source.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class SDK:
    def __getattr__(self, name):
        if name == "get_time_stamp_ns":
            return lambda: 123
        if name.endswith("controller_pose"):
            return lambda: [0., 0., 0., 0., 0., 0., 1.]
        if name.endswith("_axis"):
            return lambda: [0., 0.]
        if name.endswith(("_trigger", "_grip")):
            return lambda: .5
        return lambda: False


def test_raw_pose_and_button_mapping():
    sdk = SDK()
    sdk.get_Y_button = lambda: True
    sdk.get_A_button = lambda: True
    stamp, hands = source.read_sdk(sdk)
    assert stamp == 123
    assert hands["left"]["key_two"] and hands["right"]["key_one"]
    assert hands["right"]["pose"] == [0., 0., 0., 0., 0., 0., 1.]


def test_incoherent_or_invalid_sample_is_not_published():
    sdk = SDK()
    stamps = iter([123, 124])
    sdk.get_time_stamp_ns = lambda: next(stamps)
    assert source.read_sdk(sdk) is None
    sdk = SDK()
    sdk.get_left_controller_pose = lambda: [0.]*7
    with pytest.raises(ValueError):
        source.read_sdk(sdk)


def test_atomic_write_only_sensor_file(tmp_path):
    path = tmp_path / "sensor.json"
    source.write_packet(path, {"seq": 1})
    assert json.loads(path.read_text()) == {"seq": 1}
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError):
        source.write_packet(path, {"seq": float("nan")})
    assert json.loads(path.read_text()) == {"seq": 1}
