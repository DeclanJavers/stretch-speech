"""Stretch's arm, wrist, gripper and head through the driver's FollowJointTrajectory action
(/stretch_controller/follow_joint_trajectory: the same on the real robot and in MuJoCo).

Blocking calls, made from the state machine's thread; the node's executor (main thread)
completes the action futures.
"""

from __future__ import annotations

import threading
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = {           # our short names -> Stretch's joint names
    "lift": "joint_lift", "arm": "wrist_extension", "yaw": "joint_wrist_yaw", "pitch": "joint_wrist_pitch",
    "roll": "joint_wrist_roll", "gripper": "joint_gripper_finger_left", "head_pan": "joint_head_pan",
    "head_tilt": "joint_head_tilt",
}
GRIPPER_OPEN, GRIPPER_CLOSED = 0.5, -0.22      # Stretch gripper units (about -0.38 fully closed .. 0.56 open)


class Arm:
    def __init__(self, node, action: str = "/stretch_controller/follow_joint_trajectory",
                 joint_states: str = "/stretch/joint_states", dry_run: bool = False, stop=None):
        self.node, self.dry_run = node, dry_run
        self.stop = stop or threading.Event()             # set on shutdown: stop waiting
        self.client = ActionClient(node, FollowJointTrajectory, action)
        self.positions: dict[str, float] = {}
        self.lock = threading.Lock()
        node.create_subscription(JointState, joint_states, self._on_joints, 10)

    def _on_joints(self, msg: JointState) -> None:
        with self.lock:
            self.positions.update(zip(msg.name, msg.position))

    def position(self, short: str) -> float | None:
        with self.lock:
            return self.positions.get(JOINTS[short])

    def move(self, timeout_s: float = 20.0, duration_s: float = 2.0, **targets: float) -> dict:
        """Move the given joints (short names) and wait until done: move(lift=0.9, arm=0.3)."""
        if self.dry_run:
            time.sleep(0.5)
            return {"success": True, "reason": "dry run"}
        if not self.client.wait_for_server(timeout_sec=5.0):
            return {"success": False, "reason": "no trajectory action server (is the driver running?)"}
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [JOINTS[k] for k in targets]
        point = JointTrajectoryPoint(positions=[float(v) for v in targets.values()],
                                     time_from_start=Duration(sec=int(duration_s),
                                                              nanosec=int(duration_s % 1 * 1e9)))
        goal.trajectory.points = [point]
        for attempt in range(6):                          # the driver rejects goals while it thinks one is running
            send = self.client.send_goal_async(goal)
            if self._wait(send, 5.0) and send.result().accepted:
                break
            time.sleep(0.5)
        else:
            return {"success": False, "reason": f"arm goal rejected: {targets}"}
        result = send.result().get_result_async()
        if not self._wait(result, timeout_s):
            send.result().cancel_goal_async()
            return {"success": False, "reason": f"arm move timed out: {targets}"}
        if result.result().status != GoalStatus.STATUS_SUCCEEDED:
            # The MuJoCo driver aborts every goal after moving (it can't wait on the wheels).
            # Judge by where the joints are (the gripper may be stopped by what it holds).
            if not self._wait_joints({k: v for k, v in targets.items() if k != "gripper"}, 3.0):
                return {"success": False, "reason": f"arm move failed: {targets}"}
        return {"success": True, "reason": ""}

    def _wait_joints(self, targets: dict, timeout_s: float, tol: float = 0.03) -> bool:
        end = time.time() + timeout_s
        while time.time() < end and not self.stop.is_set():
            now = {k: self.position(k) for k in targets}
            if all(now[k] is not None and abs(now[k] - v) < tol for k, v in targets.items()):
                return True
            time.sleep(0.05)
        return False

    def _wait(self, future, timeout_s: float) -> bool:
        end = time.time() + timeout_s
        while not future.done() and time.time() < end and not self.stop.is_set():
            time.sleep(0.02)
        return future.done()
