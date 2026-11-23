# Stretch swag robot

Hello Robot Stretch working an event: it finds new people, asks what swag they want,
fetches it from its table, finds them again and hands it over.

- **`MASTER-PLAN.md`**: how the whole system works (state machine, triggers, LLM harness,
  where things run, work streams). Start here.
- **`FUTURE_WORK.md`**: ideas parked for later.

## Layout

| Folder | Runs on | What |
|---|---|---|
| `ros/stretch_swag/` | robot / VM (ROS 2 Humble) | everything that decides: state machine, event log, LLM harness, skills, plus simulation stand-ins |
| `mac-service/` | Mac | LLM server (`llama-server`) and the face service |
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
- `fake_base`: a lightweight stand-in for the robot's base (cmd_vel → odom, no physics)
- `sim_people`: five virtual visitors with different personas; they show faces to the
  "camera" and answer the robot
- `robot`: the real thing

Useful variations:
```bash
ros2 launch stretch_swag sim.launch.py auto_reply:=false   # then, in another terminal:
ros2 run stretch_swag chat_console                          # type as the visitor ('...' = pause)

ros2 launch stretch_swag sim.launch.py fake_base:=false     # drive the MuJoCo sim instead (start it first)
ros2 launch stretch_swag sim.launch.py pickup_success:=0.5  # grasps fail half the time
ros2 launch stretch_swag sim.launch.py world:=/path/crowd.json   # your own crowd (see config/world.json)
```

Watch it: `ros2 topic echo /swag/state`. Each run logs to `~/.ros/stretch_swag/run-<time>/`:
- `events.jsonl`: every state change, utterance, skill result
- `llm.jsonl`: every LLM call's prompt and output

## Tests

```bash
cd ros/stretch_swag && python3 -m pytest test -q
```
