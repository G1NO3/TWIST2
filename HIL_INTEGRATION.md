# QwenOFT HIL host

`deploy_real/server_low_level_g1_hil.py` is the opt-in host for the sibling
VLAPolicyBridge QwenOFT Actor-Critic GUI. The original BC host is unchanged.

Ported from the working `/home/jwang3617/pipette/TWIST2` checkout on
2026-09-15: its real HIL host, `hil_remote`, `hil_gravity`,
`joint_command_safety`, and dedicated copies of the body/hand wrappers.
Imports point to `g1_wrapper_hil` and `inspire_hand_hil` to isolate BC.
SIGTERM is handled through the same cleanup as Ctrl-C, including during
START/A startup. The Inspire hand driver retains its last grasp on exit
with `--hil_safety`.

The GUI supplies the Python interpreter, ONNX, URDF, NIC and hand addresses;
it uses CPU inference, Inspire hands and `--hil_safety`. START and A remain
physical operator actions. Do not run this alongside the ordinary real
host or reference adapter. It publishes `hil:unitree_remote`, its heartbeat,
and exact post-safety joint targets in Redis. It is included in the GUI's
emergency-stop sweep.

Pure tests under `tests/test_hil_*.py`, `test_joint_command_safety.py` and
`test_inspire_hil_hold.py` need no robot; `--help` checks imports without
constructing a controller. No live robot test was performed during this port.
