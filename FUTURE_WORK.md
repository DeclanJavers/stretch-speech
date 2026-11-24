# Future work

Things we know we need but aren't building yet. Add to this as we go.

## Person recognition (main blocker)

The robot has to recognize **people**, not just faces. Several states depend on it: "Looking for a new person", "Walking towards new person", "Lost new person", "Finding person again", and "Logging person in memory". The person won't always face the camera, and once the robot locks onto someone they may turn away.

### What it needs

1. **Detection.** Find every person (head and/or full body) in a crowded frame and crop them out accurately.
2. **Matching.** Compare a crop against a memory of people already seen, and decide whether it's someone known or someone new.
3. **Spatial continuity (tracking).** If a face is seen in one frame and a side profile appears in the same place in the next, assume it's the same person, and store that side profile under them.
4. **A catalog of people.** Each person gets a set of views (front, sides, back, clothing) that grows as the robot sees more of them. Matching uses the best combination it has.

### Leads to evaluate

Not yet tested by us; check accuracy and CPU speed on Stretch's computer before committing to any of these.

- **Person detection:** YOLO-family detectors (e.g. Ultralytics YOLO, person class) or RT-DETR. Small versions run on CPU at a few frames per second.
- **Face detection and embeddings:** RetinaFace / SCRFD for detection, ArcFace embeddings (the InsightFace package) for matching. These are strong when the face is visible and useless when it isn't.
- **Person re-identification (re-ID):** models trained to match whole bodies across views, mostly from clothing and build (e.g. OSNet in `torchreid`). They work from the side and back, which covers the "person turned away" case. Weak if people dress alike.
- **Multi-object tracking:** ByteTrack, BoT-SORT, DeepSORT. These give each person a track ID that persists frame to frame using position and motion, which is exactly the spatial continuity in item 3. Views captured while a track is alive can be added to that person's catalog.
- **Depth:** Stretch's head camera is an RGB-D camera, so each detection can get a 3D position. Tracking in 3D (in the map frame) is more robust than in image space, and it also gives the "where did I talk to them" location for "Finding person again".
- **Likely combination:** detector → tracker (track IDs) → per-track face and re-ID embeddings → match tracks against the catalog. Face match when the face is visible, body re-ID otherwise.

### Open questions

- Compute: what runs at a usable frame rate on Stretch's CPU, versus offloading frames to the Mac like the language server does?
- When is a match confident enough to say "already served", and when should the robot just ask ("Have I already given you something?")?
- Privacy: the catalog stores pictures of real people. Keep it in memory only, or delete it after the event.

### Cheat until it exists

Give each visitor a printed ArUco marker badge. Stretch's ROS stack already includes ArUco marker detection. The marker ID becomes the person's identity, and the head camera gives their 3D position. The rest of the state diagram can then be built and tested now. Put perception behind one interface (e.g. "list of people: id, 3D position, last seen") so real recognition can replace the badges later without touching the state machine.

## Other open items

- **Sim speed:** MuJoCo runs at ~1/3 real time in the VM because of CPU speed in the VM, not rendering. Running it natively on the Mac would fix that, without ROS in that path.
- **Speech on the robot:** run `language-server/server.py` on Stretch's computer, using its mic and speaker, with the LLM still on the Mac. No changes to the speech code needed.
- **Cancelling conversations:** Ctrl+C on the ROS node mid-conversation leaves the conversation running on the language server.
- **Cleanup:** README paths still describe the old layout; the `.gitignore` eval-results line points at `eval/` instead of `language-server/eval/`; the unused `fetch_distance_m` parameter in the behavior node.
