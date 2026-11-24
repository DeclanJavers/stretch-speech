# First test on the real Stretch (v1)

**Goal:** the whole loop on the real robot, with the cheats that are left:

| Piece | v1 on the robot |
|---|---|
| Finding / recognizing people | **ArUco badges** worn by visitors (Stretch's own marker detector) |
| Hearing visitors | **typed** by an operator (`chat_console`) |
| Robot speech | Kokoro voice on the Mac (or `espeak-ng` / `spd-say` on the robot as fallback) |
| Driving | straight lines on wheel odometry, **lidar safety stop**, slow (0.15 m/s) |
| Picking up / handover | mocked: the robot pauses at the table and "hands over" without the arm |
| Conversation, state machine, logs | the real thing |

Work through the stages in order. **Stop at the first failure**: each stage assumes the
ones before it work.

---

## 0. Before going to the lab

- [ ] **Print the badges:** `ros/stretch_swag/badges/badges_*.png`, at **100% scale** (no
      "fit to page"). Make at least 3. **Measure the black square** with a ruler; if it
      isn't 100 mm, change `length_mm` in `ros/stretch_swag/config/badges.yaml` for all ids.
- [ ] **Push the code** so it can be pulled on the robot.
- [ ] **Ask the TA / check:** the robot's IP and login, that it runs Ubuntu 22.04 + ROS 2
      Humble, the name/path of its ROS workspace, and how the head is normally pointed.
- [ ] **Network:** the robot must reach the Mac. Lab WiFi often blocks device-to-device
      traffic. Plan B: a phone hotspot both join, or an ethernet cable.

## 1. Mac (three terminals, from the repo folder)

```bash
ipconfig getifaddr en0          # the Mac's IP on this network -> use as mac_ip below
```
```bash
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_0 --alias lfm3-1.7b -c 4096 --jinja --host 0.0.0.0 --port 8081
```
```bash
python3 mac-service/face_service.py --verbose
```
```bash
cd language-server && uv run ../mac-service/speech_service.py      # Kokoro voice, port 8200
```
Allow incoming connections if macOS asks. Put the Mac (its speaker) near the robot.

## 2. Robot: install our code (once)

```bash
cd ~ && git clone <repo url> stretch-speech            # or git pull if it's already there
ln -sfn ~/stretch-speech/ros/stretch_swag ~/<robot_ws>/src/stretch_swag
cd ~/<robot_ws> && colcon build --packages-select stretch_swag --symlink-install
source install/setup.bash
cd ~/stretch-speech/ros/stretch_swag && python3 -m pytest test -q     # expect: all passed
```
No Python packages to install: our nodes use only ROS and the standard library.

From the robot, check it can reach the Mac (replace MAC_IP):
```bash
curl http://MAC_IP:8081/v1/models && curl http://MAC_IP:8100/health && curl http://MAC_IP:8200/health
```

## 3. Robot: start Stretch's own stack (one terminal each, `source` the workspace first)

```bash
stretch_robot_home.py                                         # once after power-on, if not homed
ros2 launch stretch_core stretch_driver.launch.py mode:=navigation
ros2 launch stretch_core rplidar.launch.py                    # lidar -> /scan_filtered
ros2 launch stretch_core d435i_low_resolution.launch.py       # head camera
```
- [ ] **Point the head forward, level, at face height** (pan 0, tilt about 0). Ask the TA
      for their usual way (web teleop, gamepad, or a `stretch_body` script before the driver).
- [ ] Topics exist: `ros2 topic list | grep -E "odom|scan_filtered|camera/color|stretch/cmd_vel"`
- [ ] Camera transform exists: `ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame`

## 4. Pre-flight checks (robot does not move)

**Lidar safety** (the most important check):
```bash
ros2 run stretch_swag lidar_check
```
- [ ] Nothing near the robot: **clear**. If it says BLOCKED with nothing there, the lidar is
      seeing the robot itself: note the distances it prints and tell us before going on.
- [ ] Box ~0.5 m **in front** (the direction the base drives): **BLOCKED**.
- [ ] Box ~0.5 m **behind**: clear. If front and back are swapped, rerun with
      `--ros-args -p scan_yaw_offset:=3.1416`, and pass `scan_yaw_offset:=3.1416` to the launch below.

**Badges** (start the robot stack with motion off, section 5, then):
- [ ] Hold badge 100 at ~1.5 m in front of the camera. The `badge_faces` log says
      `badge 100 at 1.5x m, bearing ~0 deg`. Check the distance with a tape measure (±15 cm is fine).
- [ ] Move it to your left: the bearing goes **positive**. To the right: negative.

**Speech:**
```bash
ros2 topic pub --once /swag/say std_msgs/msg/String "{data: '{\"text\": \"Hello, this is a test.\"}'}"
```
- [ ] The Mac speaks it (or the robot, with the fallback).

## 5. Stage A: whole loop, **no motion**

```bash
ros2 launch stretch_swag robot.launch.py mac_ip:=MAC_IP                      # motion:=false is the default
ros2 run stretch_swag chat_console                                           # second terminal: the operator
```
A visitor wearing badge 100 stands ~1.5 m in front of the robot. The operator types what
they say (Enter = they finished; `...` = a pause).

- [ ] Robot greets them, asks their name. Type `I'm Alex`.
- [ ] Ask for an item: `can I get a plushie please`. It says to wait, then "goes to the table"
      (no movement; state changes print), "picks up", comes back, calls "Alex?", finds the
      badge (`verify p001: ... match`), hands over.
- [ ] A second visitor with badge 101: decline (`no thanks, I'm good`).
- [ ] Badge 100 again: the robot does **not** start a new conversation (already served).

## 6. Stage B: **with motion** (someone on the runstop button at all times)

Clear area of about 4 × 4 m. The **table is where the robot starts** (odometry 0, 0).
```bash
ros2 launch stretch_swag robot.launch.py mac_ip:=MAC_IP motion:=true
```
- [ ] Robot spins slowly in place looking for badges (45° steps).
- [ ] Visitor at ~2 m: robot turns to face them and talks.
- [ ] After they choose: drives back to its start spot (the "table"), pauses, drives to
      ~1 m from where they stood, calls their name, hands over.
- [ ] **Safety:** step in front of it while it drives: it must stop within 0.6 m and wait.
- [ ] Visitor walks away after ordering: robot searches ~30 s, gives up, "puts the item back".

**Stop immediately** (runstop) if it drives toward someone without stopping, spins
nonstop, or moves faster than walking pace. Ctrl+C on the launch also sends a stop command.

## 7. Afterwards

- [ ] Copy the logs off the robot: `~/.ros/stretch_swag/run-*/` (`events.jsonl` = everything
      that happened, `llm.jsonl` = every prompt and reply). These are our eval data.
- [ ] Note what failed and at which stage.

## Known v1 limits (expected, not bugs)

- No obstacle avoidance: it stops for obstacles but doesn't drive around them.
- Wheel odometry drifts: after a few minutes "the table" and "where Alex stood" will be off
  by tens of cm.
- No arm motion: pickup and handover are pauses.
- The robot only knows people by badge; someone without a badge is invisible to it.
- Speech comes out of the Mac, not the robot (unless using the espeak fallback).
