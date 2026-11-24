"""Picking up a goodie bag by its rope loops, handing it over, putting it back.

The table is laid out in lanes: each item's bags stand one behind the other in a row running from
the table's front edge to the back (seen from the front: pens left, t-shirts middle, plushies
right), and at the back of each lane stands an ArUco marker on a post, above the bags' loops.
The markers tell the robot where the table is, which way it faces and where each lane is; the
bags' colour (bag_detect) tells it how far into the lane the front bag stands. Step by step:

  1. arm up and in (gripper down, open); face the table, find the markers: the line through them
     gives the table's facing, each marker its lane
  2. drive to a spot in front of the lane (open floor), face the table squarely, drive straight in
     until close (`clearance_m` from the edge: just enough room to turn in place)
  3. look down the lane: the front bag (colour) and the lane's marker, from the same images
  4. turn left 90 deg: the arm (on the robot's right) now points along the lane. From here the
     tucked-in gripper hides the bags, but the marker shows above it: re-anchor on the marker
     (fixes any slip in the turn), drive forward/back so the gripper is in line with the bag
  5. arm out above the loops, down around them, close; lift 1 cm: did it close on something?
     Yes: lift, pull in, turn away from the table. No: open, back out, measure again
  No bag seen (lighting, colour): try the lane's slots front to back (`first_slot_m`, `slot_pitch_m`).

Grasp tested in MuJoCo: tolerates ~±4 cm along the table and -4/+2 cm in reach. Geometry below is
Stretch SE3 (from the MuJoCo model) + our bag design; measure the real table and set the parameters
(handle_top_m, marker_to_edge_m, ...; see STRETCH-TEST.md).
"""

from __future__ import annotations

import math
import time

from .arm import GRIPPER_CLOSED, GRIPPER_OPEN
from .core.geometry import table_facing, wrap

# Stretch, gripper pointing down (wrist pitch -pi/2), in base_link: where the grasp point is.
GRASP_X = -0.021            # along the base's forward axis
GRASP_Y0 = -0.121           # sideways (right = negative) with the arm fully in; minus the extension
GRASP_DZ = -0.136           # height = lift + this
ARM_MIN, ARM_MAX, LIFT_MAX = 0.02, 0.50, 1.08
READY = {"lift": 0.95, "arm": 0.0, "yaw": 0.0, "pitch": -1.57, "roll": 1.57, "gripper": GRIPPER_OPEN}
LOOK_AHEAD = {"head_pan": 0.0, "head_tilt": -0.1}          # head straight ahead at face height
DEFAULT_MARKERS = {"pen": 20, "shirt": 21, "plushie": 22}


def rot(x: float, y: float, a: float) -> tuple[float, float]:
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def median(values):
    v = sorted(values)
    return v[len(v) // 2]


class Picker:
    def __init__(self, node, driver, arm, bags_in_base, markers_seen, params: dict, on_table=None):
        """bags_in_base(item, since) -> [(x, y)] that colour's bags in base_link, from an image taken
        after `since` (None if no such image yet); markers_seen(since) -> {id: {"base": (x, y),
        "odom": (x, y), "facing": yaw (odom)}} from images after `since`; on_table(facing) is told
        the table's facing (odom yaw) when the markers give it (bag_detect uses it)."""
        self.node, self.driver, self.arm = node, driver, arm
        self.bags_in_base, self.markers_seen, self.on_table = bags_in_base, markers_seen, on_table
        self.table_dir = params["table_dir"]                     # rough heading toward the table from its spot
        self.marker_ids = params.get("markers") or DEFAULT_MARKERS
        self.loops_z = params.get("handle_top_m", 0.78)          # top of the rope loops above the floor
        self.marker_to_edge = params.get("marker_to_edge_m", 0.45)
        self.clearance = params.get("clearance_m", 0.24)         # base_link -> table edge, facing it (room to turn: 0.18)
        self.stage = params.get("stage_m", 0.30)                 # first look from this much further out
        self.first_slot = params.get("first_slot_m", 0.08)       # table edge -> middle of the front bag
        self.slot_pitch = params.get("slot_pitch_m", 0.13)
        self.slots = int(params.get("slots", 3))
        self.lane_half = params.get("lane_half_width_m", 0.10)
        self.grip_depth = params.get("grip_depth_m", 0.04)       # fingers this far below the loops' top
        # where along the lane to grab, from the bag's middle (+ = toward the robot): 0 closes the
        # fingers around both loops (they must fit in the open gripper, ~9 cm on the real robot);
        # half the loops' spacing grabs just the front loop's top
        self.grasp_offset = params.get("grasp_offset_m", 0.0)
        self.markers: dict[int, tuple[float, float]] = {}        # marker id -> odom (x, y): the table, as last seen
        self.facing: float | None = None                         # table -> open side (odom yaw)
        self.taken: dict[str, tuple[float, float]] = {}          # item -> where its bag stood (lane coords)

    def log(self, text: str) -> None:
        self.node.get_logger().info(f"[pick] {text}")

    def fail(self, reason: str) -> dict:
        self.log(f"failed: {reason}")
        return {"success": False, "reason": reason}

    def where(self, label: str) -> None:
        """Log the robot's pose relative to the table (debugging the approach)."""
        pose = self.driver.pose()
        if pose is None or self.facing is None or not self.markers:
            return
        m = next(iter(self.markers.values()))
        n = (math.cos(self.facing), math.sin(self.facing))
        out = (pose[0] - m[0]) * n[0] + (pose[1] - m[1]) * n[1] - self.marker_to_edge
        self.log(f"  {label}: ({pose[0]:.3f}, {pose[1]:.3f}) yaw {math.degrees(wrap(pose[2] - self.facing)):.1f} deg "
                 f"from the table's facing, {out:.3f} m from the edge")

    def lift_at(self, z: float) -> float:
        return min(LIFT_MAX, z - GRASP_DZ)

    # ---- looking -------------------------------------------------------------------
    def sample(self, read, wait_s: float = 3.0, samples: int = 5, settle_s: float = 0.6):
        """Readings from images taken after the robot and head stopped moving: read(since) -> value
        or None. Returns up to `samples` distinct values (waits at most wait_s)."""
        time.sleep(settle_s)
        since, got, end = time.time(), [], time.time() + wait_s
        while time.time() < end and len(got) < samples:
            v = read(since)
            if v is not None and (not got or v != got[-1]):
                got.append(v)
            time.sleep(0.1)
        return got

    def look_at_table(self, pans=(0.0,), tilt: float = -0.45, need: int | None = None) -> bool:
        """Markers in view -> self.markers (odom) and self.facing. True once marker `need` (or any) is seen."""
        for pan in pans:
            self.arm.move(head_pan=pan, head_tilt=tilt)
            got = self.sample(lambda since: self.markers_seen(since) or None, samples=4)
            if not got:
                continue
            ids = {i for g in got for i in g if i in self.marker_ids.values()}
            normals, fresh = [], []
            for i in ids:
                seen = [g[i] for g in got if i in g]
                if len(seen) >= 2:
                    self.markers[i] = (median(s["odom"][0] for s in seen), median(s["odom"][1] for s in seen))
                    normals.append(math.atan2(sum(math.sin(s["facing"]) for s in seen),
                                              sum(math.cos(s["facing"]) for s in seen)))
                    fresh.append(i)
            pose = self.driver.pose()
            # one marker: its own pose gives the facing, but the line through several (seen moments
            # ago) is better
            facing = table_facing([self.markers[i] for i in fresh], normals if self.facing is None else [], pose[:2] if pose else (0.0, 0.0))
            if facing is not None:
                self.facing = facing
                if self.on_table:
                    self.on_table(facing)
            self.log(f"markers seen: {sorted(fresh)}; table faces {math.degrees(self.facing or 0):.1f} deg")
            if fresh and (need is None or need in fresh):
                return True
        return False

    def lane(self, item: str):
        """(marker odom xy, n = unit table -> open side, u = unit along the table) for item's lane."""
        m = self.markers[self.marker_ids[item]]
        n = (math.cos(self.facing), math.sin(self.facing))
        return m, n, (-n[1], n[0])

    def front_bag(self, item: str, m, n, u):
        """The lane's front bag, (along n, along u) from the marker, from images after the head
        settled at the current pose; None if not seen."""
        def read(since):
            bags = self.bags_in_base(item, since)
            pose = self.driver.pose()
            if bags is None or pose is None:
                return None
            best = None
            for bx, by in bags:
                wx, wy = rot(bx, by, pose[2])
                wx, wy = wx + pose[0] - m[0], wy + pose[1] - m[1]
                d, a = wx * n[0] + wy * n[1], wx * u[0] + wy * u[1]
                if abs(a) < self.lane_half and 0 < d < self.marker_to_edge + 0.05 and (best is None or d > best[0]):
                    best = (d, a)                      # in this lane, on the table: the one nearest the front
            return best
        got = self.sample(read)
        if len(got) < 2:
            return None
        return median(g[0] for g in got), median(g[1] for g in got)

    def marker_in_base(self, marker_id: int, tilts=(-0.45, -0.3, -0.6)):
        """The marker as seen from here, in base_link (median of a few images), or None. Head
        panned right (along the arm); a couple of tilts in case one view is blocked."""
        for tilt in tilts:
            self.arm.move(head_pan=-math.pi / 2, head_tilt=tilt)
            got = self.sample(lambda since: (self.markers_seen(since) or {}).get(marker_id, {}).get("base"),
                              samples=4, wait_s=2.5)
            if len(got) >= 2:
                return median(g[0] for g in got), median(g[1] for g in got)
        return None

    # ---- moving next to the lane ---------------------------------------------------
    def approach(self, item: str) -> dict:
        """Steps 1-3: stand facing the lane, clearance_m from the edge; returns where its front bag
        is ({"bag": (d, a) or None}) or a failure."""
        r = self.arm.move(**READY)
        if not r["success"]:
            return r
        mid = self.marker_ids.get(item)
        if mid is None:
            return self.fail(f"no lane for {item}")
        # every time: odometry drifts while the robot drives around between picks
        self.driver.turn_to(self.table_dir if self.facing is None else self.facing + math.pi)
        if not self.look_at_table(pans=(0.0, 0.5, -0.5, 1.0, -1.0), need=mid):
            if mid not in self.markers or self.facing is None:
                return self.fail(f"can't see the {item} lane's marker (id {mid})")
            self.log("can't see the markers from here: going by where they were last time")
        m, n, u = self.lane(item)
        edge = self.marker_to_edge
        stand = lambda off: (m[0] + n[0] * (edge + off), m[1] + n[1] * (edge + off))  # noqa: E731
        stage = stand(self.clearance + self.stage)
        self.log(f"{item} lane marker at ({m[0]:.2f}, {m[1]:.2f}); going in front of it ({stage[0]:.2f}, {stage[1]:.2f})")
        self.driver.docking = True
        try:
            r = self.driver.go_to(*stage, tolerance=0.05)
            if not r["success"] and r["reason"] in ("cancelled", "interrupted"):
                return r
            r = self.driver.turn_to(self.facing + math.pi, tolerance=0.02)
            if not r["success"]:
                return r
            # closer look: the table's facing and this lane's marker, then square up again
            if self.look_at_table(tilt=-0.7, need=mid):
                m, n, u = self.lane(item)
                self.driver.turn_to(self.facing + math.pi, tolerance=0.02)
            # straight in until clearance_m from the edge (no safety stop: the table is meant to be close)
            pose = self.driver.pose()
            off = (pose[0] - m[0]) * n[0] + (pose[1] - m[1]) * n[1] - edge     # base_link -> edge now
            self.where("before driving in")
            self.drive_exactly(off - self.clearance)
            self.where("driven in")
        finally:
            self.driver.docking = False
        self.arm.move(head_pan=0.0, head_tilt=-0.85)
        bag = self.front_bag(item, m, n, u)
        if bag is None:
            self.log(f"can't see a {item} bag in its lane: trying its slots blind")
        else:
            self.log(f"front {item} bag {edge - bag[0]:.3f} m in from the edge, {bag[1]:+.3f} m off the lane's middle")
        return {"success": True, "reason": "", "bag": bag}

    def side_on(self, item: str, d: float, a: float) -> dict:
        """Step 4: turn the arm side to the table and line the gripper up with the point (d, a) of
        the lane (from its marker). Returns {"reach": arm extension} or a failure."""
        m, n, u = self.lane(item)
        r = self.driver.turn_to(self.facing + 1.5 * math.pi, tolerance=0.02)   # left 90 deg: right side to the table
        if not r["success"]:
            return r
        self.where("turned")
        pose = self.driver.pose()
        target_w = (m[0] + d * n[0] + a * u[0] - pose[0], m[1] + d * n[1] + a * u[1] - pose[1])
        target = rot(*target_w, -pose[2])                         # in base_link, by odometry
        seen = self.marker_in_base(self.marker_ids[item])
        if seen is not None:                                      # anchor on the marker: what the camera sees now
            mw = rot(m[0] - pose[0], m[1] - pose[1], -pose[2])
            dx, dy = seen[0] - mw[0], seen[1] - mw[1]
            self.log(f"marker {dx * 100:+.1f} cm along, {dy * 100:+.1f} cm across from where odometry has it")
            target = (target[0] + dx, target[1])                  # along: trust the camera (depth: PnP is noisier)
        else:
            self.log("can't see the lane marker from the side: going by odometry")
        self.arm.move(head_pan=0.0, head_tilt=-0.45)
        self.drive_exactly(target[0] - GRASP_X)
        self.where("lined up")
        reach = GRASP_Y0 - target[1] - self.grasp_offset
        if not ARM_MIN <= reach <= ARM_MAX:
            return self.fail(f"bag out of reach (needs {reach:.2f} m of arm)")
        return {"success": True, "reason": "", "reach": reach}

    def leave(self) -> None:
        """Arm in, turn to face away from the table (the gripper side swings away from it), step off."""
        self.arm.move(arm=0.0)
        self.driver.turn_to(self.facing, tolerance=0.05)
        self.drive_exactly(0.25)

    # ---- the skills ------------------------------------------------------------------
    def pick(self, item: str) -> dict:
        try:
            return self._pick(item)
        finally:
            self.arm.move(**LOOK_AHEAD)                   # look for faces again

    def _pick(self, item: str) -> dict:
        tried = set()
        for attempt in (1, 2, 3):
            r = self.approach(item) if attempt == 1 or self.facing is None else self.reapproach(item)
            if not r["success"]:
                return r
            bag = r.get("bag")
            if bag is None:                                       # blind: the lane's slots, front to back
                slots = [(self.marker_to_edge - self.first_slot - k * self.slot_pitch, 0.0) for k in range(self.slots)]
                bag = next((s for s in slots if s not in tried), None)
                if bag is None:
                    break
                tried.add(bag)
            r = self.side_on(item, *bag)
            if not r["success"]:
                self.arm.move(**READY)
                self.driver.turn_to(self.facing + math.pi, tolerance=0.02)
                if "out of reach" in r["reason"]:
                    return r
                continue
            reach = r["reach"]
            self.log(f"attempt {attempt}: reaching {reach:.2f} m")
            r = self.grasp(reach)
            if not r["success"]:
                return r
            if r["holding"]:
                self.taken[item] = bag
                self.arm.move(lift=LIFT_MAX)
                self.leave()
                self.log(f"holding the {item} bag")
                return {"success": True, "reason": ""}
            # missed: open, straight up, tuck in, face the lane again and look again
            self.arm.move(gripper=GRIPPER_OPEN)
            self.arm.move(lift=self.lift_at(self.loops_z + 0.12))
            self.arm.move(arm=0.0)
            self.driver.turn_to(self.facing + math.pi, tolerance=0.02)
        self.arm.move(**READY)
        self.leave()
        return self.fail("the gripper closed on nothing")

    def reapproach(self, item: str) -> dict:
        """Already facing the lane up close (after a miss): just look again."""
        m, n, u = self.lane(item)
        self.arm.move(**READY, head_pan=0.0, head_tilt=-0.85)
        return {"success": True, "reason": "", "bag": self.front_bag(item, m, n, u)}

    def grasp(self, reach: float) -> dict:
        """Out at a safe height, just above the loops, slowly around them, close, lift 1 cm.
        {"holding": bool} or a failure."""
        top = self.loops_z
        for step in ({"lift": self.lift_at(top + 0.12)}, {"arm": reach}, {"lift": self.lift_at(top + 0.02)},
                     {"lift": self.lift_at(top - self.grip_depth)}, {"gripper": GRIPPER_CLOSED},
                     {"lift": self.lift_at(top + 0.01)}):
            r = self.arm.move(**step)
            if not r["success"]:
                return self.fail(r["reason"])
        time.sleep(0.3)
        grip = self.arm.position("gripper")
        holding = grip is None or grip > GRIPPER_CLOSED + 0.05
        self.log(f"gripper at {grip if grip is not None else float('nan'):.3f} -> {'holding' if holding else 'empty'}")
        return {"success": True, "reason": "", "holding": holding}

    def drive_exactly(self, distance: float) -> None:
        """Straight forward/back by `distance` on odometry, correcting any overshoot once."""
        start = self.driver.pose()
        if start is None or abs(distance) < 0.005:
            return
        self.driver.drive_distance(distance, speed_mps=0.05)
        p = self.driver.pose()
        moved = (p[0] - start[0]) * math.cos(start[2]) + (p[1] - start[1]) * math.sin(start[2])
        if abs(distance - moved) > 0.008:
            self.driver.drive_distance(distance - moved, speed_mps=0.03)

    def hand_over(self, person_xy, wait_s: float) -> dict:
        """Turn the arm side toward the person, hold the bag out, let go after wait_s."""
        pose = self.driver.pose()
        if pose:
            toward = math.atan2(person_xy[1] - pose[1], person_xy[0] - pose[0])
            self.driver.turn_to(toward + math.pi / 2)
        self.arm.move(arm=0.35, lift=0.95)
        time.sleep(wait_s)
        self.arm.move(gripper=GRIPPER_OPEN)
        time.sleep(1.0)
        self.arm.move(arm=0.0)
        self.arm.move(**READY, **LOOK_AHEAD)
        return {"success": True, "reason": ""}

    def put_back(self, item: str) -> dict:
        """Back where it was taken from (that spot is free: it was the lane's front bag)."""
        spot = self.taken.get(item)
        if spot is None or self.facing is None:
            return self.fail(f"don't know where the {item} bag goes")
        held = {k: v for k, v in READY.items() if k != "gripper"}
        try:
            # carried tucked in (the bag stays within the robot's footprint while it turns next to
            # the table); lift at READY height so the camera sees the lane marker over it
            self.arm.move(**held, head_pan=0.0, head_tilt=-0.45)
            m, n, u = self.lane(item)
            edge = self.marker_to_edge
            stage = (m[0] + n[0] * (edge + self.clearance + self.stage), m[1] + n[1] * (edge + self.clearance + self.stage))
            self.driver.docking = True
            try:
                self.driver.go_to(*stage, tolerance=0.05)
                self.driver.turn_to(self.facing + math.pi, tolerance=0.02)
                pose = self.driver.pose()
                off = (pose[0] - m[0]) * n[0] + (pose[1] - m[1]) * n[1] - edge
                self.drive_exactly(off - self.clearance)
            finally:
                self.driver.docking = False
            r = self.side_on(item, *spot)
            if not r["success"]:
                return r
            for step in ({"lift": LIFT_MAX}, {"arm": r["reach"]}, {"lift": self.lift_at(self.loops_z - self.grip_depth + 0.01)},
                         {"gripper": GRIPPER_OPEN}, {"lift": self.lift_at(self.loops_z + 0.12)}, {"arm": 0.0}):
                self.arm.move(**step)
            self.taken.pop(item, None)
            self.leave()
            return {"success": True, "reason": ""}
        finally:
            self.arm.move(**READY, **LOOK_AHEAD)
