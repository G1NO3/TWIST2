"""server_data_record_d435 --feature_tap (2026-11-10): every inference the policy server's feature tap publishes is
kept, and a saved episode gets features.npz with the inferences of its time span, from one server start. Local ZMQ
only; the recorder env (cv2, redis, zmq)."""
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
rec = pytest.importorskip("server_data_record_d435")

H = 8


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _send(pub, seq, t_ms, server="s1", **extra):
    q = np.full((1, 30, H), seq, np.float16)
    a = np.full((1, 30, 32), seq / 10, np.float32)
    head = dict(framework="QwenOFT", checkpoint_fp="1:ab", server=server, seq=seq, t_ms=t_ms,
                q_shape=list(q.shape), a_shape=list(a.shape), **extra)
    pub.send_multipart([b"FEAT1", json.dumps(head).encode(), q.tobytes(), (a - 1).tobytes(), a.tobytes()])


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_listener_keeps_every_inference_and_writes_the_episode_span(tmp_path):
    port = _port()
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    tap = rec.FeatureTapListener(f"127.0.0.1:{port}")
    assert _wait(lambda: (_send(pub, 0, 500, server="old"), len(tap.records) > 0)[1])   # slow joiner
    for seq, t in enumerate([1000, 1200, 1400, 1600, 5000]):
        _send(pub, seq, t)
    assert _wait(lambda: len([r for r in tap.records if r[0]["server"] == "s1"]) == 5)
    field = tap.frame_field()
    assert field["server"] == "s1" and field["seq"] == 4 and field["age_ms"] >= 0
    tap.write_episode(str(tmp_path), 400, 1700)          # the "old" server's message is in the span: newest server wins
    assert _wait(lambda: (tmp_path / "features.npz").exists())
    f = np.load(tmp_path / "features.npz")
    assert f["seq"].tolist() == [0, 1, 2, 3] and f["t_ms"].tolist() == [1000, 1200, 1400, 1600]
    assert f["queries"].shape == (4, 30, H) and f["queries"].dtype == np.float16
    assert f["queries"][2].max() == 2 and np.allclose(f["action"][3], 0.3) and np.allclose(f["base_action"][3], -0.7)
    head = json.loads(str(f["header"]))
    assert head["checkpoint_fp"] == "1:ab" and head["server"] == "s1" and "seq" not in head
    assert not list(tmp_path.glob("*.tmp.npz"))
    pub.close(0)


def test_the_noise_sigma_of_every_inference_is_kept(tmp_path):
    """residual heads (2026-11-11): each message says the noise sigma it served (the console's noise switch changes
    it between episodes); features.npz keeps it per inference, out of the episode header"""
    port = _port()
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    tap = rec.FeatureTapListener(f"127.0.0.1:{port}")
    assert _wait(lambda: (_send(pub, 0, 100, server="old"), len(tap.records) > 0)[1])
    for seq, (t, sigma) in enumerate([(1000, 0.0), (1200, 0.05), (1400, 0.05)]):
        _send(pub, seq, t, explore_sigma=sigma)
    assert _wait(lambda: len([r for r in tap.records if r[0]["server"] == "s1"]) == 3)
    tap.write_episode(str(tmp_path), 900, 1500)
    assert _wait(lambda: (tmp_path / "features.npz").exists())
    f = np.load(tmp_path / "features.npz")
    np.testing.assert_allclose(f["explore_sigma"], [0.0, 0.05, 0.05])
    assert "explore_sigma" not in json.loads(str(f["header"]))
    pub.close(0)


def test_an_episode_without_inferences_writes_nothing(tmp_path):
    tap = rec.FeatureTapListener(f"127.0.0.1:{_port()}")
    tap.write_episode(str(tmp_path), 0, 10 ** 12)
    time.sleep(0.2)
    assert not (tmp_path / "features.npz").exists() and tap.frame_field() is None
