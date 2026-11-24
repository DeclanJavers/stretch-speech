"""Picking up a goodie bag by its rope loops, handing it over, putting it back.

The bags stand on the table's front edge, each with a label carrying the item's name and an
ArUco tag (pen 200, shirt 201, plushie 202). The tag tells us where the bag is, to ~1 cm:

  1. arm up out of the way (gripper pointing down, open); face the table; find the bag's tag
  2. drive to the spot beside the bag where the arm (which reaches out of the robot's right
     side) can get to it with ~30 cm of extension, then turn so that side faces the table
  3. look at the bag from the side and correct: inch forward/back, set the arm's reach
  4. come down over the loops, close the gripper deep around them, lift, pull the arm in
  5. check: the tag rose with the gripper (or the gripper didn't close all the way)

Grasp tested in MuJoCo: holds through retracting and driving; tolerates ~±4 cm along the table
and -4/+2 cm in reach. Geometry below is Stretch SE3 (from the MuJoCo model) + our bag design;
measure the real bags and set the handle_above_tag_m / bag_depth_m parameters.
"""

from __future__ import annotations

import math
import time

from .arm import GRIPPER_CLOSED, GRIPPER_OPEN

# Stretch, gripper pointing down (wrist pitch -pi/2), in base_link: where the grasp point is.
GRASP_X = -0.021            # along the base's forward axis
GRASP_Y0 = -0.121           # sideways (right = negative) with the arm fully in; minus the extension
GRASP_DZ = -0.136           # height = lift + this
ARM_MIN, ARM_MAX, LIFT_MAX = 0.02, 0.50, 1.08
READY = {"lift": 0.95, "arm": 0.0, "yaw": 0.0, "pitch": -1.57, "roll": 1.57, "gripper": GRIPPER_OPEN}
LOOK_AHEAD = {"head_pan": 0.0, "head_tilt": -0.1}          # head straight ahead at face height


class Picker:
    def __init__(self, node, driver, arm, marker_in_base, params: dict):
        """marker_in_base(tag_id, max_age_s) -> (x, y, z) in base_link, or None."""
        self.node, self.driver, self.arm, self.marker_in_base = node, driver, arm, marker_in_base
        self.tags = params["tags"]                        # item -> tag id
        self.table_dir = params["table_dir"]              # heading from the table spot toward the table (rad)
        self.handle_above_tag = params.get("handle_above_tag_m", 0.145)
        self.bag_depth = params.get("bag_depth_m", 0.10)
        self.reach = params.get("nominal_reach_m", 0.30)
        self.slots: dict[str, dict] = {}                  # where each item was picked (to put it back)

    def log(self, text: str) -> None:
        self.node.get_logger().info(f"[pick] {text}")

    def fail(self, reason: str) -> dict:
        self.log(f"failed: {reason}")
        return {"success": False, "reason": reason}

    def find_tag(self, tag: int, wait_s: float = 3.0, pans=(0.0,)) -> tuple[float, float, float] | None:
        for pan in pans:
            if pan is not None:
                self.arm.move(head_pan=pan)
            end = time.time() + wait_s
            while time.time() < end:
                p = self.marker_in_base(tag, 0.5)
                if p is not None:
                    return p
                time.sleep(0.1)
        return None

    def bag_center(self, tag_pos) -> tuple[float, float, float]:
        """Tag (on the bag's front, facing us) -> middle of the bag, where the loops are."""
        x, y, z = tag_pos
        d = math.hypot(x, y) or 1.0
        return x + x / d * self.bag_depth / 2, y + y / d * self.bag_depth / 2, z

    def pick(self, item: str) -> dict:
        try:
            return self._pick(item)
        finally:
            self.arm.move(**LOOK_AHEAD)                   # look for faces again

    def _pick(self, item: str) -> dict:
        tag = self.tags.get(item)
        if tag is None:
            return self.fail(f"no tag id for {item}")
        r = self.arm.move(**READY, head_tilt=-0.7, head_pan=0.0)
        if not r["success"]:
            return self.fail(r["reason"])
        # 1. face the table, find the bag
        self.driver.turn_to(self.table_dir)
        seen = self.find_tag(tag, pans=(0.0, 0.5, -0.5))
        if seen is None:
            return self.fail(f"can't see the {item} bag (tag {tag})")
        bx, by, _ = self.bag_center(seen)
        pose = self.driver.pose()
        bag_w = (pose[0] + bx * math.cos(pose[2]) - by * math.sin(pose[2]),
                 pose[1] + bx * math.sin(pose[2]) + by * math.cos(pose[2]))
        # 2. the spot beside it: heading h puts the arm side (right) toward the table
        h = self.table_dir + math.pi / 2
        gx, gy = GRASP_X, GRASP_Y0 - self.reach           # grasp point in base frame at nominal reach
        spot = (bag_w[0] - (gx * math.cos(h) - gy * math.sin(h)), bag_w[1] - (gx * math.sin(h) + gy * math.cos(h)))
        self.log(f"{item} bag at ({bag_w[0]:.2f}, {bag_w[1]:.2f}); lining up at ({spot[0]:.2f}, {spot[1]:.2f})")
        self.driver.docking = True                        # close to the table: small safety distance
        try:
            self.driver.go_to(*spot, tolerance=0.06)
            self.driver.turn_to(h, tolerance=0.03)
            # 3. look from the side, correct
            self.arm.move(head_pan=-1.57, head_tilt=-0.6)
            for _ in range(3):
                seen = self.find_tag(tag, pans=(None,))
                if seen is None:
                    return self.fail(f"lost sight of the {item} bag beside it")
                bx, by, tz = self.bag_center(seen)
                dx = bx - GRASP_X
                if abs(dx) < 0.012:
                    break
                self.driver.drive_distance(max(-0.15, min(0.15, dx)), speed_mps=0.05)
                time.sleep(0.5)
        finally:
            self.driver.docking = False
        reach = GRASP_Y0 - by
        if not ARM_MIN <= reach <= ARM_MAX:
            return self.fail(f"bag out of reach (needs {reach:.2f} m of arm)")
        bar_z = tz + self.handle_above_tag
        lift_at = lambda z: min(LIFT_MAX, z - GRASP_DZ)  # noqa: E731
        self.log(f"grasp: reach {reach:.2f} m, loops at {bar_z:.2f} m")
        # 4. grasp
        for step in ({"lift": lift_at(bar_z + 0.12)}, {"arm": reach}, {"lift": lift_at(bar_z - 0.04)},
                     {"gripper": GRIPPER_CLOSED}, {"lift": lift_at(bar_z + 0.25)}, {"arm": 0.10}):
            r = self.arm.move(**step)
            if not r["success"]:
                return self.fail(r["reason"])
        # 5. check: a gripper that closed all the way has nothing in it
        time.sleep(0.5)
        grip = self.arm.position("gripper")
        self.slots[item] = {"spot": spot, "heading": h, "reach": reach, "bar_z": bar_z}
        if grip is not None:
            holding = grip > GRIPPER_CLOSED + 0.05
            self.log(f"gripper at {grip:.3f} (closed on nothing: {GRIPPER_CLOSED:.2f})")
        else:                                             # no joint states: did the tag rise?
            after = self.marker_in_base(tag, 1.0)
            holding = after is not None and after[2] > tz + 0.08
        if holding:
            self.log(f"holding the {item} bag")
            return {"success": True, "reason": ""}
        self.arm.move(gripper=GRIPPER_OPEN)
        self.arm.move(**READY)
        return self.fail("the bag didn't come up with the gripper")

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
        slot = self.slots.get(item)
        if slot is None:
            return self.fail(f"don't know where the {item} bag goes")
        self.driver.docking = True
        try:
            self.driver.go_to(*slot["spot"], tolerance=0.06)
            self.driver.turn_to(slot["heading"], tolerance=0.03)
        finally:
            self.driver.docking = False
        lift_at = lambda z: min(LIFT_MAX, z - GRASP_DZ)  # noqa: E731
        for step in ({"arm": slot["reach"]}, {"lift": lift_at(slot["bar_z"] - 0.03)}, {"gripper": GRIPPER_OPEN},
                     {"lift": lift_at(slot["bar_z"] + 0.12)}, {"arm": 0.0}):
            self.arm.move(**step)
        self.arm.move(**READY, **LOOK_AHEAD)
        return {"success": True, "reason": ""}
