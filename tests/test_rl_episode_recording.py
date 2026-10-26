"""server_data_record_d435.py --episode_status: episodes follow the RL bridge's status file."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY))
from data_utils.rl_episode_follow import RLEpisodeFollower, outcome_label  # noqa: E402


def status(t, episode=None, last=None, reason="paused", **kw):
    return dict(time=t, episode=episode, last_episode=last, reason=reason, phase="p1", paused=False, **kw)


def test_outcome_labels():
    assert outcome_label("success") == "successful"
    assert outcome_label("failure") == "unsuccessful"
    assert outcome_label("discarded") == "discarded"
    assert outcome_label("aborted") == outcome_label("aborted: camera stale") == "aborted"
    assert outcome_label("paused; X resumes SAME full-task trajectory") is None
    assert outcome_label(None) is None


def test_follow_open_close_and_late_discard(tmp_path):
    f = RLEpisodeFollower(str(tmp_path / "s.json"))
    # bridge startup: last_episode from the DB, no outcome reason -> nothing
    assert f.step(status(100, last="old", reason="paused: prepare phase and explicitly PLAY"), 100) == []
    assert f.step(status(101, episode="e1", last="old"), 101) == [("start", "e1")]
    assert f.step(status(102, episode="e1", last="old", reason="paused; X resumes"), 102) == []
    assert f.step(status(103, last="e1", reason="success"), 103) == [("save", "e1", "successful", "success")]
    assert f.step(status(104, last="e1", reason="success"), 104) == []
    # F1 after the episode closed: discard reaches the saved episode once
    assert f.step(status(105, last="e1", reason="discarded"), 105) == [("relabel", "e1", "discarded", "discarded")]
    assert f.step(status(106, last="e1", reason="discarded"), 106) == []


def test_follow_back_to_back_and_unknown_outcome(tmp_path):
    f = RLEpisodeFollower(str(tmp_path / "s.json"))
    f.step(status(1, episode="e1"), 1)
    # the terminal status was missed: e2 already open, e1 closed with an unknown outcome
    assert f.step(status(2, episode="e2", last="e1", reason="learner active"), 2) == [
        ("save", "e1", "unknown", "learner active"), ("start", "e2")]
    assert f.step(status(3, last="e2", reason="aborted: inference fault"), 3) == [
        ("save", "e2", "aborted", "aborted: inference fault")]
    # an unknown label is completed when the outcome shows up; a known one is not overwritten
    assert f.step(status(4, last="e1", reason="failure"), 4) == [("relabel", "e1", "unsuccessful", "failure")]
    assert f.step(status(5, last="e2", reason="success"), 5) == []


def test_stale_or_missing_status_aborts_but_never_starts(tmp_path):
    f = RLEpisodeFollower(str(tmp_path / "s.json"), stale_s=5.0)
    assert f.step(status(0, episode="e1"), 10) == []           # stale: no start
    assert f.step(None, 10) == []
    f.step(status(10, episode="e1"), 10)
    assert f.step(status(10, episode="e1"), 16) == [("save", "e1", "aborted", "status_stale")]
    assert f.read() is None                                     # no file yet


def test_frame_fields_and_instruction(tmp_path):
    (tmp_path / "phase_instruction.txt").write_text("Pick up the pipette.\n")
    f = RLEpisodeFollower(str(tmp_path / "s.json"), instruction_path=str(tmp_path / "phase_instruction.txt"))
    f.step(status(1, episode="e1"), 1)
    fields = f.frame_fields(status(1, episode="e1"))
    assert fields["rl_episode"] == "e1" and fields["rl_phase"] == "p1" and fields["rl_paused"] is False
    assert fields["instruction"] == "Pick up the pipette." and isinstance(fields["t_mono_ns"], int)


class _Stop(Exception):
    pass


def run_recorder(tmp_path, monkeypatch, script, stop):
    """rec.main with fake cameras/redis; script = {tick: status written by the bridge}."""
    rec = importlib.import_module("server_data_record_d435")
    status_path = tmp_path / "rlt_status.json"
    (tmp_path / "phase_instruction.txt").write_text("Aim the pipette.")
    clock = {"t": 1000.0, "tick": 0}

    def fake_time():
        return clock["t"]

    real_sleep = rec.time.sleep

    def fake_sleep(dt):
        if clock["tick"] > stop:          # main loop done: the writer's close() waits for real
            return real_sleep(dt)
        real_sleep(0.1)                # let the writer thread drain like real time would
        clock["tick"] += 1
        tick = clock["tick"]
        if tick in script:
            status_path.write_text(json.dumps(dict(script[tick], time=clock["t"])))
        if tick > stop:
            raise KeyboardInterrupt

    def redis_get(key):
        if key == "controller_data":
            raise AssertionError("controller_data must not be read with --episode_status")
        return json.dumps([0.0] * 3)

    class Pipe:
        def __init__(self):
            self.keys = []

        def get(self, key):
            self.keys.append(key)

        def execute(self):
            out, self.keys = [json.dumps([0.0] * 3) for _ in self.keys], []
            return out

    redis_ns = SimpleNamespace(ConnectionPool=lambda **kw: None,
                               Redis=lambda **kw: SimpleNamespace(ping=lambda: True, get=redis_get,
                                                                  pipeline=Pipe))
    monkeypatch.setattr(rec, "redis", redis_ns)
    monkeypatch.setattr(rec, "VisionClient", lambda **kw: SimpleNamespace(receive_process=lambda: None))
    monkeypatch.setattr(rec.time, "time", fake_time)
    monkeypatch.setattr(rec.time, "sleep", fake_sleep)
    monkeypatch.setattr(rec.cv2, "namedWindow", lambda *a: (_ for _ in ()).throw(rec.cv2.error("headless")))
    args = SimpleNamespace(data_folder=str(tmp_path / "raw"), task_name="rl", frequency=60, robot="unitree_g1",
                           robot_ip="127.0.0.1", camera_port=5555, width=8, height=6,
                           wrist_left_port=5556, wrist_right_port=None, wrist_width=8, wrist_height=6,
                           goal="g", desc="d", steps="s",
                           episode_status=str(status_path), episode_stale_s=5.0)
    rec.main(args)

    task = tmp_path / "raw" / "rl"
    return sorted(p for p in task.iterdir() if p.name.startswith("episode_"))


def test_recorder_main_follows_status(tmp_path, monkeypatch):
    """The real main loop with fake cameras/redis: every frame of each RL episode is recorded,
    the label and RL id land in data.json + rl_episode.json, controller_data is never read."""
    script = {0: status(1000, last="old", reason="paused"),
              3: status(1000, episode="e1", last="old"),
              8: status(1000, last="e1", reason="success"),
              10: status(1000, episode="e2", last="e1"),   # writer still saving e1 (<= 1 s)
              30: status(1000, last="e2", reason="failure"),
              32: status(1000, last="e2", reason="discarded")}
    episodes = run_recorder(tmp_path, monkeypatch, script, stop=34)
    assert [p.name for p in episodes] == ["episode_0000", "episode_0001"]
    d1 = json.loads((episodes[0] / "data.json").read_text())
    d2 = json.loads((episodes[1] / "data.json").read_text())
    assert d1["label"] == "successful" and d1["rl"]["episode"] == "e1" and d1["rl"]["outcome"] == "success"
    assert d2["label"] == "unsuccessful" and d2["rl"]["episode"] == "e2"
    assert 0 < len(d2["data"]) < 20      # started once e1's save finished, ran to tick 29
    # every tick while e1 was open is a frame (ticks 3..7 -> 5 frames), images on disk
    assert len(d1["data"]) == 5 and [f["idx"] for f in d1["data"]] == list(range(5))
    assert all(f["rl_episode"] == "e1" and f["instruction"] == "Aim the pipette." for f in d1["data"])
    assert (episodes[0] / d1["data"][0]["rgb"]).is_file()
    assert (episodes[0] / d1["data"][0]["rgb_wrist_left"]).is_file()
    side1 = json.loads((episodes[0] / "rl_episode.json").read_text())
    side2 = json.loads((episodes[1] / "rl_episode.json").read_text())
    assert side1["rl_episode"] == "e1" and side1["label"] == "successful"
    assert side2["rl_episode"] == "e2" and side2["label"] == "discarded"   # the late F1


def test_recorder_stop_mid_episode_saves_interrupted(tmp_path, monkeypatch):
    """GUI stop (SIGTERM -> KeyboardInterrupt) with an episode open: saved as interrupted, not successful."""
    episodes = run_recorder(tmp_path, monkeypatch, {1: status(1000, episode="e1")}, stop=5)
    data = json.loads((episodes[0] / "data.json").read_text())
    assert len(episodes) == 1 and data["label"] == "interrupted" and data["rl"]["outcome"] == "recorder_stopped"
    assert json.loads((episodes[0] / "rl_episode.json").read_text())["label"] == "interrupted"
