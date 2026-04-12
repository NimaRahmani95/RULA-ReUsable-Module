# ROS 2 RULA Ergonomic Monitor

A real-time ergonomic monitoring and robot-assisted workstation adjustment stack for ROS 2 (Jazzy). The system uses [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) with Intel RealSense depth cameras to estimate 3D skeletal pose, computes RULA (Rapid Upper Limb Assessment) scores, and drives a UR5e collaborative robot to continuously optimise the PCB workpiece height for the operator.

---

## Overview

```
Intel RealSense D-series (x3: front, right, left)
          |
          v
 [point_2D_extractor]  AlphaPose inference + 3D deprojection
          |  /multi_camera_points  (MultiCameraPoints)
          |  /operator_gesture     (String: THUMBS_UP / THUMBS_DOWN)
          v
 [rula_calculator]     3D RULA score computation
          |  /full_body_data       (BodyMsg)
          |  /left_rula_score, /right_rula_score  (Int16)
          v
 [pcb_ergonomic_assistant]  Gradient-descent Z-height optimiser
          |  RTDE  -->  UR5e robot
          |  /gui_notifications    (String)
          v
 [rula_gui]            Dark-themed dashboard (customtkinter + matplotlib)
```

The system runs a three-phase state machine:

| Phase | Description |
|---|---|
| `INIT` | Waiting for operator to appear at all three cameras |
| `RULA_OPTIMIZING` | Automatic gradient-descent adjustment of robot Z-height |
| `USER_ADJUSTMENT` | Operator fine-tunes height via thumb gestures |

---

## Features

- **Multi-camera 3D pose estimation** — Three RealSense cameras (front, right, left) combined to disambiguate shoulder flexion, upper-arm abduction, elbow angle, neck and trunk posture
- **RULA scoring** — Full implementation of RULA Table A/B pipeline using 3D joint angles from the Halpe 136-keypoint skeleton remapped to H36M-17
- **Gradient-descent optimizer** — EMA-smoothed cost function drives the robot Z-axis to the ergonomic optimum; transitions to manual control when stable or after a 60-second timeout
- **Gesture control** — Thumbs Up / Thumbs Down detected via dual-hand voting buffer with leading-edge latch; adjusts PCB height 15 mm per tap in `USER_ADJUSTMENT`
- **Real-time GUI** — Live camera feeds, colour-coded RULA score bars, phase indicator, system log, and a matplotlib trend diagram of arm scores over time
- **Audio feedback** — Procedurally generated chime plays when optimisation converges

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Robot | Universal Robots UR5e (RTDE interface) |
| Depth cameras | 3× Intel RealSense D415 / D435 / D435i |
| GPU | NVIDIA GPU with CUDA (for AlphaPose inference) |
| OS | Ubuntu 24.04 |
| ROS | ROS 2 Jazzy |

Camera placement:
- **Front** — facing the operator, centred on the workstation
- **Right** — 90° to the right of the operator, captures sagittal profile
- **Left** — 90° to the left of the operator, captures sagittal profile

---

## Software Prerequisites

### 1. ROS 2 Jazzy

```bash
# Follow the official installation guide:
# https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions
```

### 2. Intel RealSense SDK

```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update && sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev
pip install pyrealsense2
```

### 3. AlphaPose

Follow the [AlphaPose installation guide](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md).

After installation, download the required model:

```bash
# Halpe + COCO wholebody 136-keypoint model (ResNet-50, 256x192 regression)
# Place checkpoint at:
#   ~/AlphaPose/pretrained_models/multi_domain_fast50_regression_256x192.pth
# Place config at:
#   ~/AlphaPose/configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml
```

Model download: [Google Drive — AlphaPose pretrained models](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md)

### 4. Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy customtkinter matplotlib pillow
pip install ur-rtde         # UR5e RTDE control
pip install tqdm
```

### 5. cv_bridge (ROS 2)

```bash
sudo apt install ros-jazzy-cv-bridge
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Industry40Lab/RULA-ReUsable-Module.git ~/RULA-ReUsable-Module

# 2. Source ROS 2
source /opt/ros/jazzy/setup.bash

# 3. Build
cd ~/RULA-ReUsable-Module
colcon build --symlink-install

# 4. Source the workspace
source install/setup.bash
```

---

## Configuration

All robot and optimizer parameters can be set in [src/rula_calculator/config/ergonomic_assistant.yaml](src/rula_calculator/config/ergonomic_assistant.yaml):

```yaml
pcb_ergonomic_assistant:
  ros__parameters:
    robot_ip:              "192.168.0.100"   # UR5e IP address
    movement_cooldown_sec: 1.5               # seconds between optimizer moves
    z_min_limit:           0.35              # metres — lowest allowed TCP Z
    z_max_limit:           0.65              # metres — highest allowed TCP Z
```

Camera serial numbers / device names are passed as launch arguments (see Usage).

To change AlphaPose model paths, edit [src/point_2D_extractor/point_2D_extractor/point_2D.py](src/point_2D_extractor/point_2D_extractor/point_2D.py) lines 64–65.

---

## Usage

### Option A — Full stack (single command)

```bash
ros2 launch rula_gui ergonomic_stack.launch.py \
    front_id:=<front-camera-serial> \
    right_id:=<right-camera-serial> \
    left_id:=<left-camera-serial>
```

Find camera serial numbers with:
```bash
rs-enumerate-devices | grep Serial
```

### Option B — Launch nodes individually

**Terminal 1 — Pose extraction (all 3 cameras)**
```bash
ros2 launch point_2D_extractor video_audio_system_launch.py \
    front_id:=<serial> right_id:=<serial> left_id:=<serial>
```

**Terminal 2 — RULA calculator + ergonomic assistant**
```bash
ros2 run rula_calculator rula_calculator &
ros2 run rula_calculator pcb_ergonomic_assistant \
    --ros-args -p robot_ip:=192.168.0.100
```

**Terminal 3 — GUI**
```bash
ros2 run rula_gui rulaGui
```

### Gesture control (USER_ADJUSTMENT phase only)

| Gesture | Action |
|---|---|
| Thumbs Up (hold 3+ frames) | Raise PCB 15 mm |
| Thumbs Down (hold 3+ frames) | Lower PCB 15 mm |

---

## ROS 2 Topics

| Topic | Type | Publisher | Description |
|---|---|---|---|
| `/multi_camera_points` | `body_data/MultiCameraPoints` | point_2D_extractor | 3D keypoints from all cameras |
| `/operator_gesture` | `std_msgs/String` | point_2D_extractor | `THUMBS_UP` or `THUMBS_DOWN` |
| `/front_frame_2D` | `sensor_msgs/Image` | point_2D_extractor | Annotated front camera feed |
| `/right_frame_2D` | `sensor_msgs/Image` | point_2D_extractor | Annotated right camera feed |
| `/left_frame_2D` | `sensor_msgs/Image` | point_2D_extractor | Annotated left camera feed |
| `/full_body_data` | `body_data/BodyMsg` | rula_calculator | Full RULA assessment output |
| `/left_rula_score` | `std_msgs/Int16` | rula_calculator | Left side RULA score (1–9) |
| `/right_rula_score` | `std_msgs/Int16` | rula_calculator | Right side RULA score (1–9) |
| `/gui_notifications` | `std_msgs/String` | pcb_ergonomic_assistant | Phase/action log messages |

---

## Custom Messages

### `body_data/BodyMsg`

| Field | Type | Description |
|---|---|---|
| `right_arm_up` / `left_arm_up` | `float32` | Upper arm flexion angle (degrees) |
| `right_low_angle` / `left_low_angle` | `float32` | Lower arm (elbow) angle (degrees) |
| `neck_angle` / `trunk_angle` | `float32` | Forward flexion from upright (degrees) |
| `right_shoulder` / `left_shoulder` | `int32` | Shoulder raised flag (0/1) |
| `right_up_abduction` / `left_up_abduction` | `int32` | Upper arm abduction flag (0/1) |
| `right_low_abduction` / `left_low_abduction` | `int32` | Lower arm abduction flag (0/1) |
| `neck_twist` / `neck_bending` / `side_bending` | `int32` | Neck/trunk deviation flags (0/1) |
| `right_rula_score` / `left_rula_score` | `int16` | Final RULA score (1–9) |
| `up_arm_score_right/left` | `int16` | Upper arm RULA sub-score |
| `lower_arm_score_right/left` | `int16` | Lower arm RULA sub-score |
| `neck_score` / `trunk_score` | `int16` | Neck / trunk RULA sub-score |
| `right` / `left` | `bool` | Side camera visibility flag |

### `body_data/MultiCameraPoints`

| Field | Type | Description |
|---|---|---|
| `header` | `std_msgs/Header` | Timestamp |
| `front_points` | `float32[]` | Flattened `[X, Y, Z, ...]` — 136 keypoints × 3 (front camera) |
| `right_points` | `float32[]` | Same format — right camera |
| `left_points` | `float32[]` | Same format — left camera |

---

## Package Structure

```
src/
├── body_data/                    # Custom ROS 2 message definitions
│   └── msg/
│       ├── BodyMsg.msg
│       └── MultiCameraPoints.msg
├── point_2D_extractor/           # AlphaPose + RealSense depth integration
│   ├── point_2D_extractor/
│   │   └── point_2D.py           # Pose inference, 3D deprojection, gesture detection
│   └── launch/
│       └── video_audio_system_launch.py
├── rula_calculator/              # RULA scoring + UR5e optimiser
│   ├── rula_calculator/
│   │   ├── rula_calculator.py            # 3D RULA computation node
│   │   ├── pcb_ergonomic_assistant.py    # Gradient-descent Z-height optimiser
│   │   └── proactive_rtde_controller.py  # Proportional RTDE controller (alternative)
│   └── config/
│       └── ergonomic_assistant.yaml
└── rula_gui/                     # Dashboard GUI
    ├── rula_gui/
    │   └── rulaGui.py            # customtkinter + matplotlib dashboard
    └── launch/
        ├── rula_run.launch.py
        └── ergonomic_stack.launch.py     # Full-stack launcher
```

---

## RULA Methodology

RULA scores joints on the following scales:

| Body part | Score range | Inputs |
|---|---|---|
| Upper arm | 1–4 + adjustments | Flexion angle, shoulder raise, abduction |
| Lower arm | 1–2 + adjustments | Elbow angle, arm crossing midline |
| Wrist | (static — score 1) | |
| Neck | 1–3 + adjustments | Forward flexion, twist, lateral bend |
| Trunk | 1–4 + adjustments | Forward lean, side bend |
| Legs | 1–2 | Support quality (parameter) |

The final RULA score (1–9) is read from Table C (combination of upper-limb and neck/trunk scores). Scores 1–2 indicate acceptable posture; 7+ require immediate corrective action.

Key geometry notes:
- All angles are computed in true 3D space from metric (metre-scale) joint coordinates
- `points2angle(A, B, C)` returns the angle at vertex B (interior angle of triangle ABC)
- Neck/trunk upright posture yields ~180° from the raw angle function (anti-parallel vectors); the code applies `180° − raw` to convert to forward-flexion degrees
- Abduction is detected both angularly (>40° shoulder-to-hip angle) and metrically (wrist >25 cm from spine centreline)

---

## Optimizer Details

The `pcb_ergonomic_assistant` node runs a gradient-descent loop on robot TCP Z-height:

```
cost = weight_upper * max(0, upper_arm_angle - 20)^2
     + weight_lower * lower_arm_deviation(elbow_angle)^2

z_offset = -lr * dCost/dZ
```

Convergence criteria (first to fire wins):
1. **Stability window** — 70% of the last 12 cycles have `|z_offset| < 8 mm`
2. **Plateau detection** — sum of `|z_offset|` over last 15 cycles < 8 mm total
3. **Hard timeout** — 60 seconds elapsed since entering `RULA_OPTIMIZING`

EMA smoothing (`alpha = 0.25`) reduces sensor noise before computing the gradient.

---

## Known Limitations

- AlphaPose hand-keypoint confidence scores for Halpe 136 commonly fall in the 0.10–0.30 range; the gesture confidence threshold is set to 0.10 accordingly
- Depth holes in the RealSense image are patched with a 5×5 median filter; occluded joints fall back to (0, 0, 0) in 3D
- The RTDE controller requires a direct Ethernet connection to the UR5e (not WiFi)
- The GUI is sized for a 2560×1600 display; scale the `ctk.CTk` window geometry for smaller screens

---

## Acknowledgement

This work was carried out within the framework of the project **ARISE**, funded by the **European Union** under Grant Agreement **No. 101135784**.

---
## License

MIT License — see [LICENSE](LICENSE) for details.

Third-party components:
- [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) — Apache 2.0
- [librealsense](https://github.com/IntelRealSense/librealsense) — Apache 2.0
- [ur-rtde](https://gitlab.com/sdurobotics/ur_rtde) — MIT
- [hri_msgs](https://github.com/ros4hri/hri_msgs) — Apache 2.0
- [hri_actions_msgs](https://github.com/ros4hri/hri_actions_msgs) — BSD

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{rula_ergo_ros2,
  title  = {ROS 2 RULA Ergonomic Monitor},
  year   = {2026},
  url    = {https://github.com/NimaRahmani95/RULA-ReUsable-Module}
}
```
