"""Exercise the real host's B publication without importing robot hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "deploy_real/server_low_level_g1_real.py"


def host_methods(writes):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "RealTimePolicyController")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("__init__", "_publish_motion_start")]
    module = ast.Module(body=methods, type_ignores=[])
    class BeforeHardware(Exception):
        pass
    def stop_before_hardware(*args):
        raise BeforeHardware()
    redis = SimpleNamespace(set=lambda k, v: writes.append((k, v)), pipeline=lambda: None)
    ns = {"redis": SimpleNamespace(Redis=lambda **kw: redis),
          "Config": stop_before_hardware, "key_pressed": lambda keys, mask: bool(keys & mask)}
    exec(compile(module, str(SOURCE), "exec"), ns)
    return ns, BeforeHardware, redis


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_b_never_publishes_motion_start_and_clears_old_signal(enabled):
    writes = []
    ns, stop, redis = host_methods(writes)
    host = SimpleNamespace()
    with pytest.raises(stop):
        ns["__init__"](host, "unused.onnx", "unused.yaml", motion_start_button_enabled=enabled)
    assert writes == ([] if enabled else [("motion_start_signal", "0")])
    host.env = SimpleNamespace(controller_mapping={"B": 0x0200})
    for keys in (0, 0x0200, 0x0200 | 0x0400, 0x0100):
        ns["_publish_motion_start"](host, keys)
        assert writes[-1] == ("motion_start_signal", "1" if enabled and keys & 0x0200 else "0")


def test_host_cli_and_loop_wire_the_option_without_changing_start_a():
    source = SOURCE.read_text()
    assert "parser.add_argument('--disable_motion_start_button'" in source
    assert "motion_start_button_enabled=not args.disable_motion_start_button" in source
    assert "self._publish_motion_start(remote_keys)" in source
    assert "self.env.move_to_default_pos()" in source
    assert "self.env.default_pos_state()" in source
