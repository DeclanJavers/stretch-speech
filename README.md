# Stretch swag robot

Hello Robot Stretch working an event: it finds new people, asks what swag they want,
fetches it from its table, finds them again and hands it over.

- **`MASTER-PLAN.md`**: how the whole system works (state machine, triggers, LLM harness,
  where things run, work streams). Start here.
- **`FUTURE_WORK.md`**: ideas parked for later.
- **`STRETCH-TEST.md`**: step-by-step checklist for running v1 on the real robot.

## Layout

| Folder | Runs on | What |
|---|---|---|
| `ros/stretch_swag/` | robot / VM (ROS 2 Humble) | everything that decides: state machine, event log, LLM harness, skills, plus simulation stand-ins |
| `mac-service/` | Mac | LLM server (`llama-server`), face service, speech service (robot voice) |
| `language-server/` | Mac | the earlier speech module (Whisper + Kokoro + `talk_to_person`), used for the progress-report demo |

## Run the whole loop in simulation

**Mac** (see `mac-service/README.md`):
```bash
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_0 --alias lfm3-1.7b -c 4096 --jinja --host 0.0.0.0 --port 8081
python3 mac-service/face_service.py
```

**VM**, once (and after adding files):
```bash
ln -sfn ~/stretch-speech/ros/stretch_swag ~/ament_ws/src/stretch_swag
cd ~/ament_ws && colcon build --packages-select stretch_swag --symlink-install
```

**VM**, each time:
```bash
source ~/ament_ws/install/setup.bash
ros2 launch stretch_swag sim.launch.py
```

This starts:
- `fake_base`: a lightweight stand-in for the robot's base (cmd_vel → odom, no physics, no camera)
- `sim_people`: five virtual visitors with different personas; they show faces to the
  "camera" and answer the robot
- `robot`: the real thing

Useful variations:
```bash
ros2 launch stretch_swag sim.launch.py auto_reply:=true    # visitors answer by themselves
ros2 run stretch_swag chat_console                          # (default) type as the visitor ('...' = pause)

ros2 launch stretch_swag sim.launch.py pickup_success:=0.5  # grasps fail half the time
ros2 launch stretch_swag sim.launch.py world:=/path/crowd.json   # your own crowd (see config/world.json)
```

### With MuJoCo: real camera and lidar (`mujoco.launch.py`)

Visitors are cutout photos standing in the MuJoCo scene and turning to face the robot. The robot
recognizes them by their faces through the simulated head camera (`face_detect`, the same chain as
on the real robot), and its safety stop and search use the simulated lidar. Physics runs faster
than real time; the conversation doesn't. Run it in the **VM's desktop**:
```bash
ros2 launch stretch_swag mujoco.launch.py                    # 2x speed; type the visitors' lines in chat_console
ros2 launch stretch_swag mujoco.launch.py speed:=3
ros2 launch stretch_swag mujoco.launch.py auto_reply:=true   # visitors answer by themselves
ros2 launch stretch_swag mujoco.launch.py viewer:=false      # no window (also works over SSH)
ros2 run rqt_image_view rqt_image_view /swag/debug           # camera + everyone it has met (faces, names, items)
ros2 launch stretch_swag mujoco.launch.py pick_mode:=mock    # skip the arm (faster)
```
What happens: the robot turns in place looking; the moment it sees a new face (even 5 m away)
it walks up to ~1.1 m, checks up close that it's really someone new, and talks. Asked for an
item, it drives to the table behind its start spot, finds the lanes by the ArUco markers at the back
of the table, faces that item's lane, drives in close, measures the front bag (its colour), turns
its arm side to the table, re-checks its position on the lane's marker, grabs the rope loops with
the gripper pointing down, carries it over, hands it out (opens the gripper) and goes looking again.

Face recognition: YuNet + SFace (OpenCV, CPU, no PyTorch). Download the
models once: `bash ros/stretch_swag/models/get_models.sh`. The cutouts need photos for faces:
put `Jeff.jpg`, `Maria.jpg`, ... (names from `config/world.json`) in `ros/stretch_swag/sim_faces/`;
without one, a person gets a drawn face (which YuNet does detect, but all drawn faces look alike).
```
Dress rehearsal for the real robot (MuJoCo plays the hardware, the real launch file on top):
```bash
ros2 launch stretch_swag mujoco.launch.py stack:=false
ros2 launch stretch_swag robot.launch.py mac_ip:=192.168.64.1 motion:=true pick_mode:=arm wander:=true arena:=1.0,-3.0,6.0,3.0
```
Why the stock MuJoCo sim crawled: it traced the 360-ray lidar against ~1M mesh vertices on
every 2 ms physics step and synced status between processes on every step (max ~0.5x real
time). `sim/mujoco_world.py` computes the lidar at 15 Hz and syncs at 50 Hz (~10x headroom)
without editing Stretch's packages.

Watch it: `ros2 topic echo /swag/state`. Each run logs to `~/.ros/stretch_swag/run-<time>/`:
- `events.jsonl`: every state change, utterance, skill result
- `llm.jsonl`: every LLM call's prompt and output

## Run on the real Stretch (v1)

See `STRETCH-TEST.md`. In short, with Stretch's driver, lidar and camera running:
```bash
ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac IP>                 # motion off
ros2 launch stretch_swag robot.launch.py mac_ip:=<Mac IP> motion:=true    # motion on
ros2 run stretch_swag chat_console                                        # type what visitors say
```
Picking (`pick_mode:=arm`) needs the goodie-bag table set up and measured: STRETCH-TEST.md, section 6c. Also on the Mac:
`cd language-server && uv run ../mac-service/speech_service.py` for the robot's voice.

## Tests

```bash
cd ros/stretch_swag && python3 -m pytest test -q          # unit tests
python3 eval/run_eval.py                                  # LLM test set (needs the Mac's llama-server)
```
