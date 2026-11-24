"""Small pure functions shared by the robot node and the perception adapters (easy to test)."""

from __future__ import annotations

import math


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def scan_obstacle(ranges, angle_min: float, angle_increment: float, stop_distance: float,
                  half_width: float = math.radians(60), yaw_offset: float = 0.0,
                  min_range: float = 0.15) -> float | None:
    """Closest lidar return within stop_distance inside the forward cone, or None.

    yaw_offset: angle of the robot's forward direction in the lidar's frame (rad). On Stretch the
    lidar may be mounted rotated: check it once (STRETCH-TEST.md) and set this parameter.
    min_range: ignore returns closer than this (the robot's own body, dust on the lidar).
    """
    closest = None
    for i, r in enumerate(ranges):
        if not (min_range < r < stop_distance) or math.isinf(r) or math.isnan(r):
            continue
        angle = wrap(angle_min + i * angle_increment - yaw_offset)
        if abs(angle) <= half_width and (closest is None or r < closest):
            closest = r
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
