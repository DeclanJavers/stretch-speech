"""Small pure functions shared by the robot node and the perception adapters (easy to test)."""

from __future__ import annotations

import math


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def scan_obstacle(ranges, angle_min: float, angle_increment: float, stop_distance: float,
                  half_width: float = 0.35, yaw_offset: float = 0.0,
                  min_range: float = 0.15) -> float | None:
    """Distance to the closest lidar return in the robot's path, or None if the path is clear.

    The path is a corridor straight ahead: stop_distance long, half_width metres either side of
    the robot's centre line (Stretch's base is ~0.34 m wide, so 0.35 leaves ~18 cm each side).
    Things beside the path (a table corner, a person standing next to it) don't stop the robot.
    yaw_offset: angle of the robot's forward direction in the lidar's frame (rad); Stretch's
    lidar is mounted rotated 180 deg (the robot node reads it from TF).
    min_range: ignore returns closer than this (the robot's own body, dust on the lidar).
    """
    closest = None
    for i, r in enumerate(ranges):
        if not (min_range < r < stop_distance + half_width) or math.isinf(r) or math.isnan(r):
            continue
        a = angle_min + i * angle_increment - yaw_offset
        ahead, side = r * math.cos(a), r * math.sin(a)
        if 0 < ahead < stop_distance and abs(side) < half_width and (closest is None or ahead < closest):
            closest = ahead
    return closest


def badge_embedding(badge_id: int, dim: int = 128) -> list[float]:
    """A stand-in "face" for an ArUco badge: one-hot on the badge id, so the same badge always
    matches itself (cosine 1) and never another badge (cosine 0) in the Mac face service."""
    v = [0.0] * dim
    v[badge_id % dim] = 1.0
    return v


def rotate(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate vector v by unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def transform_point(translation, rotation, point) -> tuple[float, float, float]:
    """Apply a TF transform (translation xyz, rotation quaternion xyzw) to a point."""
    r = rotate(rotation, point)
    return (r[0] + translation[0], r[1] + translation[1], r[2] + translation[2])


def base_to_world(robot_pose: tuple[float, float, float], px: float, py: float) -> tuple[float, float]:
    """A point in the robot's base frame (x forward, y left) -> odom/map frame, given the robot (x, y, yaw)."""
    rx, ry, yaw = robot_pose
    return rx + math.cos(yaw) * px - math.sin(yaw) * py, ry + math.sin(yaw) * px + math.cos(yaw) * py


def free_distance_points(points, origin: tuple[float, float], heading: float, half_width: float = 0.35,
                         max_distance: float = 10.0) -> float:
    """How far the robot can drive from origin along heading before something in `points`
    (world x, y) is inside its path, a corridor half_width wide on each side."""
    ox, oy = origin
    c, s = math.cos(heading), math.sin(heading)
    free = max_distance
    for x, y in points:
        dx, dy = x - ox, y - oy
        ahead, side = c * dx + s * dy, -s * dx + c * dy
        if ahead > 0 and abs(side) < half_width:
            free = min(free, ahead)
    return free


def free_distance_scan(ranges, angle_min: float, angle_increment: float, heading_rel: float,
                       half_width: float = 0.35, yaw_offset: float = 0.0, min_range: float = 0.15,
                       max_distance: float = 10.0) -> float:
    """Same as free_distance_points, from a lidar scan; heading_rel is relative to robot forward."""
    points = []
    for i, r in enumerate(ranges):
        if not (min_range < r < max_distance) or math.isinf(r) or math.isnan(r):
            continue
        a = angle_min + i * angle_increment - yaw_offset
        points.append((r * math.cos(a), r * math.sin(a)))
    return free_distance_points(points, (0.0, 0.0), heading_rel, half_width, max_distance)


def arena_reach(origin: tuple[float, float], heading: float, arena: tuple[float, float, float, float]) -> float:
    """Distance along the ray from origin at heading to where it leaves the arena rectangle
    (xmin, ymin, xmax, ymax). From outside, the distance to where it leaves after crossing it;
    0 if the ray misses the arena."""
    ox, oy = origin
    dx, dy = math.cos(heading), math.sin(heading)
    t_in, t_out = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, arena[0], arena[2]), (oy, dy, arena[1], arena[3])):
        if abs(d) < 1e-9:
            if not lo <= o <= hi:
                return 0.0
            continue
        t1, t2 = (lo - o) / d, (hi - o) / d
        t_in, t_out = max(t_in, min(t1, t2)), min(t_out, max(t1, t2))
    return t_out if t_out > t_in else 0.0
