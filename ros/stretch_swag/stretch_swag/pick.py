"""Picking up a goodie bag by its rope loops, handing it over, putting it back.

The table is laid out in lanes: each item's bags stand one behind the other in a row running from
the table's front edge to the back (seen from the front: pens left, t-shirts middle, plushies
right), and at the back of each lane stands an ArUco marker on a post, above the bags' loops.
The markers tell the robot where the table is, which way it faces and where each lane is; the
bags stand at calibrated slots along each lane. Step by step:

  1. arm up and in (gripper down, open); face the table, find the markers: the line through them
     gives the table's facing, each marker its lane
  2. square up to the table and drive straight in until close (`clearance_m` from the edge: just
     enough room to turn in place), wherever along the table the robot happens to be
  3. turn left 90 deg: the arm (on the robot's right) now points into the table. Drive along the
     table until the gripper is in line with the lane, anchored on the lane's marker (it shows
     over the tucked-in gripper); re-check after a long drive
  4. the bags stand at calibrated slots (`first_slot_m` from the edge, then every `slot_pitch_m`);
     the robot counts what it took from each lane (restart = full lanes): reach for the next slot
  5. arm out above the loops, down around them, close all the way; lift 1 cm: did the fingers stop
     on something? Yes: lift, pull in, turn away from the table. No: open, pull in, look again
  Two misses at a slot: try the next one back. Everything is positioned from the markers only.
  Height limit: the lift reaches ~1.1 m, so the loops' top must be below ~0.93 m (low table).

Grasp tested in MuJoCo: tolerates ~±4 cm along the table and -4/+2 cm in reach. Geometry below is
Stretch SE3 (from the MuJoCo model) + our bag design; measure the real table and set the parameters
(handle_top_m, marker_to_edge_m, ...; see STRETCH-TEST.md).
"""

from __future__ import annotations

import math
import time

from .arm import GRIPPER_OPEN
from .core.geometry import table_facing, wrap

# Stretch, gripper pointing down (wrist pitch -pi/2), in base_link: where the grasp point is.
GRASP_X = -0.021            # along the base's forward axis
GRASP_Y0 = -0.121           # sideways (right = negative) with the arm fully in; minus the extension
GRASP_DZ = -0.136           # height = lift + this
ARM_MIN, ARM_MAX, LIFT_MAX = 0.02, 0.50, 1.08
# How far the gripper opens to go around the loops: ~13 cm between the fingers in MuJoCo (0.5 would
# be 17 cm, and a finger catches the next bag's loop in the lane). The real gripper opens ~9-12 cm at
# most; the driver clamps the command to that.
GRASP_OPEN = 0.37
# Close all the way (the driver clamps to the gripper's limit: ~-0.37 rad real, ~-0.21 MuJoCo). With
# nothing between them the fingers get there; on a bag's paper / rope they stop around 0 or above.
GRASP_CLOSE = -0.4
HOLDING_ABOVE = -0.12        # finger angle (rad): stopped above this = something between the fingers
READY = {"lift": 0.95, "arm": 0.0, "yaw": 0.0, "pitch": -1.57, "roll": 1.57, "gripper": GRASP_OPEN}
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
        self.first_slot = params.get("first_slot_m", 0.08)       # table edge -> middle of the front bag
        self.slot_pitch = params.get("slot_pitch_m", 0.13)
        self.slots = int(params.get("slots", 3))
        self.lane_half = params.get("lane_half_width_m", 0.10)
        # The grasp: fingers close this far below the loops' top, centred grasp_offset_m from the bag's
        # middle toward the robot. Sim: 0.04 / 0 (around both loops, pinching the bag's top edge).
        # Real bags: tune at the table with `ros2 run stretch_swag pick_test` (STRETCH-TEST.md 6c).
        self.grip_depth = params.get("grip_depth_m", 0.04)
        self.grasp_offset = params.get("grasp_offset_m", 0.0)
        top_grip = LIFT_MAX + GRASP_DZ                              # highest the fingers can be
        if self.loops_z - self.grip_depth > top_grip - 0.05:
            node.get_logger().warn(
                f"[pick] the loops' top at {self.loops_z:.2f} m is too high for Stretch's lift: it can grip "
                f"there but not lift the bag clear of the table. Keep the loops' top below "
                f"{top_grip - 0.05 + self.grip_depth:.2f} m (a lower table, or shorter bags).")
        self.markers: dict[int, tuple[float, float]] = {}        # marker id -> odom (x, y): the table, as last seen
        self.facing: float | None = None                         # table -> open side (odom yaw)
        self.next_slot: dict[str, int] = {}                     # item -> its lane's first slot with a bag
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

    def marker_in_base(self, marker_id: int, toward=None, tilts=(-0.45, -0.3, -0.6)):
        """The marker as seen from here, in base_link (median of a few images), or None. Head aimed
        at `toward` (base_link x, y: where odometry expects it; default: straight right, along the
        arm); a couple of tilts in case one view is blocked."""
        pan = math.atan2(toward[1], toward[0]) if toward else -math.pi / 2
        for tilt in tilts:
            self.arm.move(head_pan=pan, head_tilt=tilt)
            got = self.sample(lambda since: (self.markers_seen(since) or {}).get(marker_id, {}).get("base"),
                              samples=4, wait_s=2.5)
            if len(got) >= 2:
                return median(g[0] for g in got), median(g[1] for g in got)
        return None

    def in_base(self, wx: float, wy: float):
        """An odom point in base_link, by odometry."""
        pose = self.driver.wait_for_pose(3.0)
        return rot(wx - pose[0], wy - pose[1], -pose[2])

    # ---- moving next to the lane ---------------------------------------------------
    def dock(self, item: str) -> dict:
        """From wherever it stands in front of the table: find the markers, face the table
        squarely and drive straight in to clearance_m from the edge (no detour to the lane: the
        robot gets in line with it later, driving along the table)."""
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
        r = self.driver.turn_to(self.facing + math.pi, tolerance=0.02)     # square to the table
        if not r["success"]:
            return r
        pose = self.driver.pose()
        off = (pose[0] - m[0]) * n[0] + (pose[1] - m[1]) * n[1] - self.marker_to_edge   # base_link -> edge now
        self.where("before driving in")
        self.drive_exactly(off - self.clearance)        # no safety stop: the table is meant to be close
        self.where("driven in")
        return {"success": True, "reason": ""}

    def approach(self, item: str) -> dict:
        """Steps 1-4: up close to the table, arm side toward it, gripper in line with the lane."""
        r = self.arm.move(**READY)
        if not r["success"]:
            return r
        r = self.dock(item)
        if not r["success"]:
            return r
        return self.side_on(item, 0.0)

    def side_on(self, item: str, a: float) -> dict:
        """Turn the arm side to the table (left 90 deg) and drive along it until the gripper is in
        line with the lane, `a` m off its middle, anchored on the lane's marker (seen over the
        tucked-in gripper); re-checked after a long drive."""
        m, n, u = self.lane(item)
        r = self.driver.turn_to(self.facing + 1.5 * math.pi, tolerance=0.02)
        if not r["success"]:
            return r
        self.where("turned")
        point = (m[0] + a * u[0], m[1] + a * u[1])
        for check in (1, 2):
            expect = self.in_base(*m)
            seen = self.marker_in_base(self.marker_ids[item], toward=expect)
            along = self.in_base(*point)[0]
            if seen is not None:
                dx = seen[0] - expect[0]
                self.log(f"marker {dx * 100:+.1f} cm along, {(seen[1] - expect[1]) * 100:+.1f} cm across "
                         f"from where odometry has it")
                along += dx                                       # along: trust the camera
            else:
                self.log("can't see the lane marker from the side: going by odometry")
            move = along - GRASP_X
            self.drive_exactly(move)
            if abs(move) < 0.05 or seen is None:                  # short move: nothing to re-check
                break
        self.where("in line with the lane")
        return {"success": True, "reason": ""}

    def look_down_lane(self, item: str):
        """Side-on to the table: lower the lift so the tucked-in gripper is out of the camera's way,
        look straight down the lane and find its front bag by colour: (d, a) from the marker, or None."""
        m, n, u = self.lane(item)
        self.arm.move(lift=max(0.40, self.loops_z - 0.25), head_pan=-math.pi / 2, head_tilt=-0.85)
        bag = self.front_bag(item, m, n, u)
        self.arm.move(lift=READY["lift"])
        if bag is None:
            self.log(f"can't see a {item} bag in its lane")
        else:
            self.log(f"front {item} bag {self.marker_to_edge - bag[0]:.3f} m in from the edge, "
                     f"{bag[1]:+.3f} m off the lane's middle")
        return bag

    def reach_for(self, item: str, d: float, a: float) -> dict:
        """Already side-on at the lane: nudge along the table so the gripper is over the point
        (d, a) of the lane; returns {"reach": arm extension} or a failure."""
        m, n, u = self.lane(item)
        point = (m[0] + d * n[0] + a * u[0], m[1] + d * n[1] + a * u[1])
        x, y = self.in_base(*point)
        if abs(x - GRASP_X) > 0.01:
            self.drive_exactly(x - GRASP_X)
            x, y = self.in_base(*point)
        reach = GRASP_Y0 - y - self.grasp_offset
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
        r = self.approach(item)
        if not r["success"]:
            return r
        # no colours: the bags stand at calibrated slots; take the lane's first slot not yet emptied
        k = self.next_slot.get(item, 0)
        for attempt in (1, 2, 3):
            if k >= self.slots:
                break
            bag = (self.marker_to_edge - self.first_slot - k * self.slot_pitch, 0.0)
            self.log(f"slot {k + 1}: {self.first_slot + k * self.slot_pitch:.2f} m in from the edge")
            r = self.reach_for(item, *bag)
            if not r["success"]:
                self.arm.move(**READY)
                self.leave()
                return r
            self.log(f"attempt {attempt}: reaching {r['reach']:.2f} m")
            g = self.grasp(r["reach"])
            if not g["success"]:
                return g
            if g["holding"]:
                self.taken[item] = bag
                self.next_slot[item] = k + 1
                self.arm.move(lift=LIFT_MAX)
                self.leave()
                self.log(f"holding the {item} bag")
                return {"success": True, "reason": ""}
            # missed: open, straight up, tuck in; the 2nd miss moves on to the next slot (already empty?)
            if attempt >= 2:
                k += 1
            self.arm.move(gripper=GRASP_OPEN)
            self.arm.move(lift=self.lift_at(self.loops_z + 0.12))
            self.arm.move(arm=0.0)
        self.arm.move(**READY)
        self.leave()
        return self.fail("the gripper closed on nothing")

    def grasp(self, reach: float) -> dict:
        """Out at a safe height, just above the loops, slowly around them, close, lift 1 cm.
        {"holding": bool} or a failure."""
        top = self.loops_z
        for step in ({"lift": self.lift_at(top + 0.12)}, {"arm": reach}, {"lift": self.lift_at(top + 0.02)},
                     {"lift": self.lift_at(top - self.grip_depth)}, {"gripper": GRASP_CLOSE},
                     {"lift": self.lift_at(top + 0.01)}):
            r = self.arm.move(**step)
            if not r["success"]:
                return self.fail(r["reason"])
        time.sleep(0.3)
        grip = self.arm.position("gripper")
        holding = grip is None or grip > HOLDING_ABOVE
        self.log(f"gripper at {grip if grip is not None else float('nan'):.3f} -> {'holding' if holding else 'empty'}")
        return {"success": True, "reason": "", "holding": holding}

    def drive_exactly(self, distance: float) -> None:
        """Straight forward/back by `distance` on odometry, correcting any overshoot once."""
        start = self.driver.pose()
        if start is None or abs(distance) < 0.005:
            return
        self.driver.drive_distance(distance, speed_mps=0.10 if abs(distance) > 0.2 else 0.05)
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
            r = self.dock(item)
            if not r["success"]:
                return r
            r = self.side_on(item, spot[1])
            if not r["success"]:
                return r
            r = self.reach_for(item, *spot)
            if not r["success"]:
                return r
            for step in ({"lift": LIFT_MAX}, {"arm": r["reach"]}, {"lift": self.lift_at(self.loops_z - self.grip_depth + 0.01)},
                         {"gripper": GRIPPER_OPEN}, {"lift": self.lift_at(self.loops_z + 0.12)}, {"arm": 0.0}):
                self.arm.move(**step)
            self.taken.pop(item, None)
            self.next_slot[item] = max(0, self.next_slot.get(item, 1) - 1)
            self.leave()
            return {"success": True, "reason": ""}
        finally:
            self.arm.move(**READY, **LOOK_AHEAD)
