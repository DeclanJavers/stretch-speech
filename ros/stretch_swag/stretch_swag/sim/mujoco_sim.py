"""Stretch's MuJoCo ROS driver, unchanged, but with our faster physics loop and visible people.

Started by `ros2 launch stretch_swag mujoco.launch.py` with the stock driver's parameters
(mode, use_mujoco_viewer, use_cameras, ...). Everything that differs lives in
sim/mujoco_world.py; this file only plugs it in:

  * the MuJoCo process uses FastServer / FastPassiveServer (see mujoco_world)
  * only the head colour camera is rendered (the other four cameras cost time and we don't use them)

Settings come from environment variables (SWAG_SIM_SPEED, SWAG_WORLD, ...; see mujoco_world),
which the launch file sets.
"""

import os
import time


def patch_driver() -> None:
    import stretch_mujoco.stretch_mujoco_simulator as sms
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    from . import mujoco_world

    sms.MujocoServer = mujoco_world.FastServer
    sms.MujocoServerPassive = mujoco_world.FastPassiveServer

    original_init = sms.StretchMujocoSimulator.__init__

    def init(self, scene_xml_path=None, model=None, camera_hz=30, cameras_to_use=(), *args, **kwargs):
        cameras = [StretchCameras.cam_d435i_rgb] if cameras_to_use else []
        hz = float(os.environ.get("SWAG_CAMERA_HZ", camera_hz))
        original_init(self, scene_xml_path, model, hz, cameras, *args, **kwargs)

    sms.StretchMujocoSimulator.__init__ = init

    # The stock trajectory server ends every arm/head goal by waiting for the wheels to stop,
    # which this stretch_mujoco can't do for velocity-controlled wheels (it raises, and the goal
    # is aborted after the arm already moved). Wheels aren't part of our goals: skip that wait.
    original_wait = sms.StretchMujocoSimulator.wait_while_is_moving

    def wait_while_is_moving(self, actuator, *args, **kwargs):
        if getattr(actuator, "name", "") in ("left_wheel_vel", "right_wheel_vel"):
            return True
        return original_wait(self, actuator, *args, **kwargs)

    sms.StretchMujocoSimulator.wait_while_is_moving = wait_while_is_moving

    # It also waits up to 5 s for every joint to reach its target. A gripper closed on a bag's
    # loops never does, so every grasp step burned the full 5 s. Gripper: 1 s; others: 3 s.
    # Waits over 1.5 s are printed, to see what's slow.
    original_setpoint = sms.StretchMujocoSimulator.wait_until_at_setpoint

    def wait_until_at_setpoint(self, actuator, timeout=5.0, position_tolerance=0.05):
        name = getattr(actuator, "name", str(actuator))
        t0 = time.time()
        ok = original_setpoint(self, actuator, timeout=1.0 if name == "gripper" else 3.0,
                               position_tolerance=position_tolerance)
        if time.time() - t0 > 1.5:
            print(f"[swag] arm: {name} took {time.time() - t0:.1f} s ({'reached' if ok else 'gave up'})", flush=True)
        return ok

    sms.StretchMujocoSimulator.wait_until_at_setpoint = wait_until_at_setpoint

    # When cmd_vel stops (twist timeout), the driver spends the next second calling
    # move_by(base_translate, 0.0) to hold the base still; the simulator reads 0 as "negative"
    # and drives backward at full speed until the base has moved. Every stop ended with a
    # 5-7 cm lurch backward (and a few degrees of turn). A zero move is a stop.
    from stretch_mujoco.enums.actuators import Actuators

    original_move_by = sms.StretchMujocoSimulator.move_by

    def move_by(self, actuator, pos, *args, **kwargs):
        if actuator in (Actuators.base_translate, Actuators.base_rotate) and pos == 0.0:
            return self.set_base_velocity(0.0, 0.0)
        return original_move_by(self, actuator, pos, *args, **kwargs)

    sms.StretchMujocoSimulator.move_by = move_by


def main():
    patch_driver()
    from stretch_mujoco_driver.stretch_mujoco_driver import main as driver_main
    driver_main()


if __name__ == "__main__":
    main()
