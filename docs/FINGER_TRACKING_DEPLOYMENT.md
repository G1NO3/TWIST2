# Finger Tracking Deployment Guide

End-to-end workflow for deploying PICO 4 Ultra finger tracking with the Unitree G1 robot and Inspire RH56DFTP hands.

---

## System Overview

```
PICO 4 Ultra VR Headset
  │  Body pose (SMPLX) + Hand tracking (26 joints/hand, OpenXR)
  │  via XRobotStreamer (WiFi)
  ▼
Host Computer (workstation)
  ├─ Terminal 1: teleop.sh        ← gmr conda env
  │    xrobot_teleop_to_robot_w_hand.py
  │      ├─ GMR retargeting (SMPLX → 35D robot obs + neck)
  │      ├─ PicoFingerTracker (26-joint hand → 6-DOF Inspire angles)
  │      └─ Publishes to Redis (localhost:6379)
  │
  ├─ Terminal 2: sim2real.sh      ← twist2 conda env
  │    server_low_level_g1_real.py
  │      ├─ Reads actions from Redis
  │      ├─ Runs ONNX policy at 50 Hz → motor commands
  │      └─ Sends hand angles via Modbus TCP
  │
  └─ Terminal 3 (optional): data_record.sh
       server_data_record.py
         └─ Records state/action pairs + RGB
  │
  ▼ (Ethernet 192.168.123.0/24)
Unitree G1 Robot (192.168.123.164)
  ├─ Left Inspire Hand  (192.168.123.210:6000, Modbus TCP)
  └─ Right Inspire Hand (192.168.123.211:6000, Modbus TCP)
```

---

## Prerequisites

| Item | Details |
|------|---------|
| PICO 4 Ultra | With hand tracking enabled in settings |
| XRobotToolkit | Installed on PICO + PC service running |
| Host computer | GPU (CUDA), Ethernet port, WiFi (for PICO) |
| Conda envs | `gmr` (Python 3.10) and `twist2` (Python 3.8) |
| Redis | Running on localhost:6379 |
| G1 robot | In debug mode (L2+R2 on remote) |
| Inspire hands | Connected to robot internal network |
| ONNX checkpoint | `assets/ckpts/twist2_1017_20k.onnx` |

---

## Step-by-Step Deployment

### 1. Network Setup (Host Computer)

Connect the host computer to the G1 robot via Ethernet:

```bash
# Set the workstation IP on the Ethernet interface connected to the robot
# Replace <eth_interface> with your actual interface (e.g., enp45s0, eno1)
sudo ip addr add 192.168.123.222/24 dev <eth_interface>

# Verify connectivity
ping 192.168.123.164   # G1 robot
ping 192.168.123.210   # Left Inspire hand
ping 192.168.123.211   # Right Inspire hand
```

Find your Ethernet interface name with `ip link show` — look for the wired interface connected to the robot.

### 2. PICO 4 Ultra Setup

1. **Enable hand tracking** in PICO settings:
   - Settings → Lab/Experimental → Hand Tracking → ON

2. **Connect PICO to the same WiFi network** as the host computer
   - The XRobotToolkit PC service must be reachable from the headset

3. **Launch XRobotToolkit** on the PICO headset
   - This streams body pose (SMPLX format) and hand tracking (26 joints per hand, OpenXR format) to the PC service

4. **Data format from PICO** (per hand):
   - 26 joints × 7 values: position (x, y, z) + quaternion (qx, qy, qz, qw)
   - Joint layout: Palm(0), Wrist(1), Thumb(2-5), Index(6-10), Middle(11-15), Ring(16-20), Pinky(21-25)

### 3. Start Redis (Host Computer)

```bash
# Check if Redis is already running
redis-cli ping
# Should return: PONG

# If not running:
redis-server &
```

### 4. Put Robot in Debug Mode

On the G1 robot's physical remote control:
- Press **L2 + R2** simultaneously to enter debug mode
- The robot should be standing and ready to receive commands

### 5. Start the Low-Level Controller (Terminal 1)

This runs the ONNX policy and sends motor/hand commands to the robot.

```bash
cd ~/TWIST2

# Edit sim2real.sh to set your network interface:
#   net=enp45s0   ← change to your Ethernet interface name

bash sim2real.sh
```

What `sim2real.sh` does:
```bash
source ~/miniconda3/bin/activate twist2
python deploy_real/server_low_level_g1_real.py \
    --policy assets/ckpts/twist2_1017_20k.onnx \
    --net enp45s0 \
    --device cuda \
    --use_hand \
    --hand_type inspire
```

This will:
- Connect to the G1 robot via Unitree SDK2
- Connect to both Inspire hands via Modbus TCP (192.168.123.210/211:6000)
- Initialize hands to open position
- Start reading actions from Redis and running the policy at 50 Hz

**Wait for**: "Inspire hand connections established" and the control loop to start before proceeding.

### 6. Start Teleoperation with Finger Tracking (Terminal 2)

```bash
cd ~/TWIST2

# Option A: Edit teleop.sh to enable finger tracking
# Uncomment --finger_tracking at the bottom of teleop.sh, then:
bash teleop.sh

# Option B: Run directly with finger tracking enabled
source ~/miniconda3/bin/activate gmr
cd deploy_real
python xrobot_teleop_to_robot_w_hand.py \
    --robot unitree_g1 \
    --hand_type inspire \
    --finger_tracking \
    --actual_human_height 1.6 \
    --redis_ip localhost \
    --target_fps 100 \
    --measure_fps 1
```

**Key flags:**
| Flag | Description |
|------|-------------|
| `--finger_tracking` | **Required.** Enables Pico hand tracking → Inspire hand conversion |
| `--hand_type inspire` | Select Inspire hand DOF mapping (6-DOF, 0-1000 range) |
| `--actual_human_height` | Operator height in meters (use slightly less than actual — PICO underestimates) |
| `--smooth` | Optional: enable body motion smoothing (sliding window average) |
| `--pinch_mode` | Optional: only index+thumb close (incompatible with finger tracking) |
| `--neck_retarget_scale` | Amplification for neck tracking (default 1.5) |

### 7. Operating in Finger Tracking Mode

When `--finger_tracking` is enabled, the system uses a **keyboard-based state machine** instead of PICO controller buttons for start/stop (since the operator's hands are being tracked, not holding controllers):

**Keyboard Controls (in the teleop terminal):**
| Key | Action |
|-----|--------|
| `s` or `Enter` | Cycle state: idle → teleop → pause → teleop ... |
| `q` | Exit program (interpolates to default pose first) |
| `e` | Emergency stop (kills sim2real.sh immediately) |

**Workflow:**
1. Put on the PICO headset and ensure XRobotToolkit is streaming
2. In the teleop terminal, press `s` or `Enter` to start teleop
3. Move your body — the robot mirrors your motion
4. Move your fingers — the robot's Inspire hands mirror your hand poses
5. Press `s` to pause, `s` again to resume
6. Press `q` to exit gracefully

**Hand tracking details:**
- Each finger's curl is computed from inter-bone angles (straight=0, closed=1000)
- Thumb has separate bend and rotation (opposition) channels
- EMA smoothing (alpha=0.3) reduces jitter
- Deadzone (0.05) prevents noise when hands are open
- Curl gain (1.5) amplifies closure for more responsive feel

### 8. Data Recording (Optional, Terminal 3)

To record demonstrations while teleoperating:

```bash
cd ~/TWIST2
bash data_record.sh
```

Recording is triggered by the **PICO left controller Y button**. Each episode saves:
- RGB images (from vision stream via ZMQ port 5555)
- Body state/action (29 DOF positions, velocities, quaternion, angular velocity)
- Hand state/action (6-DOF per hand: angles, temperatures, forces)
- Timestamps (millisecond precision)

---

## Redis Data Flow Reference

### Teleop → Low-Level Server (published by teleop at ~100 Hz)

| Redis Key | Shape | Description |
|-----------|-------|-------------|
| `action_body_unitree_g1_with_hands` | 35D | [vx, vy, z, roll, pitch, yaw_angvel, 29_dof_pos] |
| `action_hand_left_unitree_g1_with_hands` | 6D | [pinky, ring, middle, index, thumb_bend, thumb_rot] (0-1000) |
| `action_hand_right_unitree_g1_with_hands` | 6D | Same as left |
| `action_neck_unitree_g1_with_hands` | 2D | [neck_yaw, neck_pitch] |
| `t_action` | int | Timestamp in milliseconds |
| `controller_data` | JSON | Button/axis states from PICO |

### Low-Level Server → Redis (published at ~50 Hz)

| Redis Key | Shape | Description |
|-----------|-------|-------------|
| `state_body_unitree_g1_with_hands` | 34D | [ang_vel(3), roll_pitch(2), dof_pos(29)] |
| `state_hand_left_unitree_g1_with_hands` | 6D | Actual hand angles |
| `state_hand_right_unitree_g1_with_hands` | 6D | Actual hand angles |
| `t_state` | int | Timestamp in milliseconds |

---

## Inspire Hand DOF Mapping

Each hand has 6 DOF. Values range from **0** (fully open) to **1000** (fully closed).

| Index | Joint | Notes |
|-------|-------|-------|
| 0 | Pinky | 0=open, 1000=closed |
| 1 | Ring | 0=open, 1000=closed |
| 2 | Middle | 0=open, 1000=closed |
| 3 | Index | 0=open, 1000=closed |
| 4 | Thumb bend | 0=open, 1000=closed |
| 5 | Thumb rotation | 0=neutral, 1000=full opposition |

Default poses (from `deploy_real/data_utils/params.py`):
- **Open**: `[0, 0, 0, 0, 0, 0]`
- **Closed**: `[1000, 1000, 1000, 1000, 800, 800]`

---

## Troubleshooting

### PICO hand tracking not working
- Ensure hand tracking is enabled in PICO settings (Lab/Experimental section)
- Verify XRobotToolkit is running and connected
- Check that the teleop script prints hand data shapes (should be 26×7 or 27×7)
- If `PicoFingerTracker` prints "Unexpected hand data shape", check XRobotToolkit version

### Inspire hands not responding
- Verify network: `ping 192.168.123.210` and `ping 192.168.123.211`
- Check Modbus connection: the low-level server should print "Left/Right hand connected"
- Ensure `--hand_type inspire` is passed to **both** `teleop.sh` and `sim2real.sh`
- If hands are stuck, power cycle them and restart `sim2real.sh` (re_init clears errors)

### Hands jittery or unresponsive
- Increase `smoothing_alpha` (default 0.3, try 0.5-0.7 for more smoothing)
- Increase `curl_deadzone` (default 0.05, try 0.1 to filter more noise)
- Decrease `curl_gain` (default 1.5, try 1.0 for less sensitivity)
- These are set in `StateMachine.__init__` when `PicoFingerTracker` is created

### Redis connection issues
- Ensure Redis is running: `redis-cli ping` → `PONG`
- Check `--redis_ip` matches between teleop and low-level server (both should be `localhost` when running on the same machine)

### Emergency stop
- **Keyboard** (finger tracking mode): Press `e` in the teleop terminal
- **Controller** (button mode): Press left controller axis click
- Both methods kill the `sim2real.sh` process immediately
- After emergency stop, you must restart `sim2real.sh` and the teleop script

---

## Key File Reference

| File | Description |
|------|-------------|
| `teleop.sh` | Launch teleop (gmr env) |
| `sim2real.sh` | Launch low-level controller (twist2 env) |
| `deploy_real/xrobot_teleop_to_robot_w_hand.py` | Main teleop script with state machine |
| `deploy_real/data_utils/finger_tracking.py` | Pico hand → Inspire angle conversion |
| `deploy_real/robot_control/inspire_hand_wrapper.py` | Modbus TCP controller for Inspire hands |
| `deploy_real/data_utils/params.py` | Default hand poses and mimic obs config |
| `deploy_real/server_low_level_g1_real.py` | Low-level ONNX policy + motor/hand control |
| `deploy_real/robot_control/configs/g1.yaml` | Robot joint config (KP/KD, scaling, etc.) |
