import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "pico_gmr_check", Path(__file__).resolve().parents[1] / "deploy_real/check_pico_gmr.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.mark.parametrize("script", [
    "pico_intervention_source.py", "pico_gmr_intervention_source.py", "check_pico_gmr.py",
])
def test_import_only_preflight_is_not_an_sdk_reader(script):
    path = str(Path(check.__file__).with_name(script))
    argv = [b"/env/bin/python3.10", path.encode(), b"--preflight"]
    assert check.pico_reader_script(argv) is None
    assert check.pico_reader_script(argv[:-1]) == script
    assert check.pico_reader_script(argv[:-1] + [b"--inspect"]) == script
    assert check.pico_reader_script(argv[:-1] + [b"--preflight-fake"]) == script
    # Unknown checkout implementations must not bypass the guard.
    assert check.pico_reader_script(["python", "/other/" + script, "--preflight"]) == script


def test_legacy_teleop_and_ambiguous_invocations_are_not_exempt():
    legacy = "xrobot_teleop_to_robot_w_hand.py"
    assert check.pico_reader_script(["python", legacy, "--preflight"]) == legacy
    path = str(Path(check.__file__).with_name("pico_gmr_intervention_source.py"))
    assert check.pico_reader_script(["python", "--preflight", path]) == Path(path).name
    assert check.pico_reader_script(["bash", "-c", path, "--preflight"]) == Path(path).name
    assert check.pico_reader_script(["RoboticsServiceProcess"]) is None
    assert check.pico_reader_script([]) is None


def test_proc_scan_allows_preflight_but_still_refuses_real_reader(tmp_path):
    path = str(Path(check.__file__).with_name("pico_gmr_intervention_source.py"))
    proc = tmp_path / "123456789"
    proc.mkdir()
    cmdline = proc / "cmdline"
    cmdline.write_bytes(b"python\0" + path.encode() + b"\0--preflight\0")
    check.assert_no_other_pico_readers(tmp_path)
    cmdline.write_bytes(b"python\0" + path.encode() + b"\0")
    with pytest.raises(RuntimeError, match=r"PID 123456789 \(pico_gmr_intervention_source.py\)"):
        check.assert_no_other_pico_readers(tmp_path)
    # A process exiting during enumeration is harmless.
    cmdline.unlink()
    check.assert_no_other_pico_readers(tmp_path)


def body_fixture():
    body = np.zeros((24, 7))
    body[:, 6] = 1
    for i, name in enumerate(check.BODY_NAMES):
        x = -.2 if name.startswith("Left") else .2 if name.startswith("Right") else 0
        y = 1.
        if any(s in name for s in ("Knee",)):
            y = .55
        if any(s in name for s in ("Ankle", "Foot")):
            y = .1
        if any(s in name for s in ("Spine", "Shoulder", "Collar", "Neck", "Head")):
            y = 1.5
        if any(s in name for s in ("Elbow", "Wrist", "Hand")):
            y = 1.2
            x *= 2
        body[i, :3] = [x, y, .05]
    return body


def test_displacements_are_unscaled_and_baseline_stays_fixed():
    d = check.Displacements()
    d.measure("left", [1, 2, 3])
    assert d.measure("left", [1, 2.1, 3])["norm_mm"] == pytest.approx(100)
    assert d.measure("left", [1, 2.2, 3])["xyz_mm"] == pytest.approx([0, 200, 0])
    assert check.Displacements().measure("left", [1, 2.2, 3])["norm_mm"] == 0


def test_invalid_body_is_not_retargeted():
    with pytest.raises(ValueError, match="Body unavailable"):
        check.human_data(None, lambda x: x, ["Pelvis"])
    b = body_fixture()
    b[20, 3:] = 0
    with pytest.raises(ValueError, match="Left_Wrist"):
        check.human_data(b, lambda x: x, ["Left_Wrist"])
    assert not check.valid_pose([0] * 7)
    assert not check.valid_pose([float("nan"), 0, 0, 0, 0, 0, 1])


def test_bad_raw_values_can_still_be_logged():
    import json
    out = check.json_safe({"pose": [float("nan"), float("inf"), 1.]})
    assert json.loads(json.dumps(out, allow_nan=False)) == {"pose": [None, None, 1.]}


def test_body_path_does_not_require_valid_controller_pose():
    class SDK:
        def get_time_stamp_ns(self): return 100
        def get_left_controller_pose(self): return [0] * 7
        def get_right_controller_pose(self): return [0] * 7
        def is_body_data_available(self): return True
        def get_body_joints_pose(self): return body_fixture().tolist()
        def get_body_timestamp_ns(self): return 0
        def get_body_joints_timestamp(self): return [100] * 24
    _, controllers, body, clock = check.read_sample(SDK())
    assert not check.valid_pose(controllers["left"])
    assert clock[1] == 100
    assert "Pelvis" in check.human_data(body, lambda x: x, ["Pelvis"])
    sdk = SDK()
    times = iter([100, 101])
    sdk.get_time_stamp_ns = lambda: next(times)
    assert check.read_sample(sdk) is None


def test_real_gmr_conversion_and_ik_offline():
    # This test initializes no SDK and imports no robot publisher.
    pytest.importorskip("mink")
    from general_motion_retargeting import GeneralMotionRetargeting, XRobotStreamer
    streamer = XRobotStreamer.__new__(XRobotStreamer)
    r = GeneralMotionRetargeting("xrobot", "unitree_g1", actual_human_height=1.6, verbose=False)
    body = body_fixture()
    human = check.human_data(body, streamer.coordinate_transform_unity_data, r.human_scale_table)
    first = r.retarget(human, offset_to_ground=True)
    body[20:22, 1] += .1
    moved = check.human_data(body, streamer.coordinate_transform_unity_data, r.human_scale_table)
    assert np.linalg.norm(np.array(moved["Left_Wrist"][0]) - human["Left_Wrist"][0]) == pytest.approx(.1)
    second = r.retarget(moved, offset_to_ground=True)
    assert first.shape == second.shape == (36,)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert np.linalg.norm(second-first) > 1e-5


def test_no_robot_output_imports():
    import ast
    tree = ast.parse(Path(check.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(n.name for n in node.names)
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(any(s in m for s in ("redis", "unitree_sdk", "robot_control", "cyclonedds"))
                   for m in modules)
