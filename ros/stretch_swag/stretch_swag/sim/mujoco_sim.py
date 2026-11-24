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


def main():
    patch_driver()
    from stretch_mujoco_driver.stretch_mujoco_driver import main as driver_main
    driver_main()


if __name__ == "__main__":
    main()
