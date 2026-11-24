"""MuJoCo side of the swag sim: a faster physics loop, and people you can see.

Runs inside the MuJoCo process that Stretch's driver spawns (no ROS here). Imported by
`mujoco_sim`, which plugs these classes into the stock `stretch_mujoco_driver`, so the ROS
topics are exactly the real robot's (/odom, /scan_filtered, /camera/color/image_raw, TF).

Why the stock sim is slow: it traces the 360-ray lidar against ~1M mesh vertices on every
2 ms physics step (2.1 ms of each step), and copies the robot's status between processes
on every step (0.9 ms). We compute the lidar at 15 Hz and sync status at 50 Hz instead;
physics then runs ~10x faster than real time, so it can be paced at `SWAG_SIM_SPEED`.

People are cardboard cutouts (full-body photos, sim_faces/<name>.png) that always turn to
face the robot. Their positions come from config/world.json and, while the sim
runs, from the file sim_people writes (`SWAG_PEOPLE_FILE`), so people can walk around.

Settings (environment variables, set by `mujoco_sim`):
    SWAG_SIM_SPEED      physics speed vs real time (default 1.0)
    SWAG_WORLD          crowd file (config/world.json)
    SWAG_PEOPLE_FILE    live positions written by sim_people (optional)
    SWAG_CARD_DIR       where to write the cutout pictures
    SWAG_FACES_DIR      photos for the cutouts' faces: <name>.jpg/.png (e.g. Jeff.jpg), or "photo" in
                        world.json. Without a photo a person gets a drawn face.
    SWAG_BAG_JITTER     test: move each bag up to this far (m) sideways in its lane, 1.5 cm along it (SWAG_BAG_SEED)
    SWAG_CAMERA_SCALE   head camera resolution vs the stock 424x240 (default 1.5 = 640x360;
                        the real robot's high-resolution mode is 1280x720 = 3.0)
"""

import json
import math
import os
import pathlib
import time

import mujoco
import numpy as np

CLOTHES = [(52, 101, 164), (170, 60, 60), (60, 140, 80), (200, 140, 40), (110, 70, 150),
           (40, 130, 140), (150, 90, 60), (90, 90, 90), (190, 80, 140), (70, 110, 60)]
SKIN = [(241, 194, 160), (198, 134, 96), (141, 85, 54), (224, 172, 105), (255, 219, 172)]


# ---- head camera resolution ----------------------------------------------------------------
def patch_head_camera_resolution() -> None:
    """Render the head camera bigger than the stock 424x240 (faces are too small past ~1.5 m
    at that size). Must run in both processes: the driver writes camera_info from these
    settings, the MuJoCo process renders with them."""
    import dataclasses
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    k = float(os.environ.get("SWAG_CAMERA_SCALE", "1.5"))
    original = StretchCameras.initial_camera_settings
    if k == 1.0 or getattr(original.fget, "_swag", False):
        return

    def settings(cam):
        s = original.fget(cam)
        if cam in (StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth):
            s = dataclasses.replace(s, width=round(s.width * k), height=round(s.height * k),
                                    focal=(s.focal[0] * k, s.focal[1] * k))
        return s

    prop = property(settings)
    prop.fget._swag = True
    type.__setattr__(StretchCameras, "initial_camera_settings", prop)


# ---- people pictures -------------------------------------------------------------------
FACE_WIDTH_M = 0.14                    # detected face width; face_detect assumes the same to get distance


def find_photo(name: str, explicit: str | None) -> str | None:
    if explicit and os.path.isfile(explicit):
        return explicit
    folder = os.environ.get("SWAG_FACES_DIR") or str(pathlib.Path(__file__).resolve().parents[2] / "sim_faces")
    for ext in ("png", "jpg", "jpeg"):
        for stem in (name, name.lower()):
            path = os.path.join(folder, f"{stem}.{ext}")
            if os.path.isfile(path):
                return path
    return None


def photo_card(path: str, photo: str):
    """A full-body photo as the cutout, scaled so the face is FACE_WIDTH_M wide (a real person's
    size). Returns (png, width_m, height_m), or None if no face is found in the photo."""
    import cv2

    img = cv2.imread(photo)
    if img is None:
        return None
    models = pathlib.Path(__file__).resolve().parents[2] / "models"
    det = cv2.FaceDetectorYN.create(str(models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.7)
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None:
        print(f"[swag] no face found in {photo}: using a drawn person", flush=True)
        return None
    m_per_px = FACE_WIDTH_M / max(faces, key=lambda f: f[2])[2]
    scale = min(4.0, 1000 * m_per_px)                       # texture ~1000 px per metre (upsampled if small)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, img)
    return path, img.shape[1] / scale * m_per_px, img.shape[0] / scale * m_per_px


def drawn_card(path: str, index: int):
    """No photo: a simple drawn person (its face is detected, but all drawn faces look alike)."""
    from PIL import Image, ImageDraw

    px_per_m = 1000
    w, h = int(0.5 * px_per_m), int(1.7 * px_per_m)
    img = Image.new("RGB", (w, h), (215, 205, 185))
    d = ImageDraw.Draw(img)
    shirt, skin = CLOTHES[index % len(CLOTHES)], SKIN[index % len(SKIN)]
    cx = w // 2
    y = lambda m: int(m * px_per_m)                                   # metres from the top of the card
    d.rounded_rectangle([cx - y(0.20), y(0.30), cx + y(0.20), y(0.95)], radius=y(0.06), fill=shirt)
    d.rectangle([cx - y(0.17), y(0.95), cx - y(0.02), y(1.68)], fill=(45, 50, 70))
    d.rectangle([cx + y(0.02), y(0.95), cx + y(0.17), y(1.68)], fill=(45, 50, 70))
    d.rectangle([cx - y(0.035), y(0.25), cx + y(0.035), y(0.31)], fill=skin)
    d.ellipse([cx - y(0.08), y(0.04), cx + y(0.08), y(0.26)], fill=skin)
    d.ellipse([cx - y(0.045), y(0.11), cx - y(0.02), y(0.135)], fill=(30, 30, 30))
    d.ellipse([cx + y(0.02), y(0.11), cx + y(0.045), y(0.135)], fill=(30, 30, 30))
    d.arc([cx - y(0.045), y(0.15), cx + y(0.045), y(0.21)], 20, 160, fill=(120, 40, 40), width=5)
    img.save(path)
    return path, 0.5, 1.7


# ---- scene ------------------------------------------------------------------------------
def load_world(path: str | None) -> list[dict]:
    if not path:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("stretch_swag"), "config", "world.json")
    return json.loads(open(path).read())["people"]


def build_model(scene_xml: str, people: list[dict], card_dir: str) -> mujoco.MjModel:
    """The stock scene plus one cutout per person, as mocap bodies we can move and turn."""
    card_dir = os.path.abspath(card_dir)
    os.makedirs(card_dir, exist_ok=True)
    spec = mujoco.MjSpec.from_file(scene_xml)
    dock = spec.find_body("link_docking_station")             # stock scene: the dock sits 1 m behind the
    if dock is not None:                                        # start (= our table spot); move it out of the way
        dock.pos = [-8.0, 0.0, dock.pos[2]]
    table = spec.find_body("table")                            # the swag table: behind the start spot, front
    if table is not None:                                      # edge 0.5 m behind it, long side along y
        table.pos = [TABLE_CENTER[0], TABLE_CENTER[1], table.pos[2]]
        table.quat = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    for name in ("object1", "object2"):                         # the stock scene's two blocks: out of the way
        obj = spec.find_body(name)
        if obj is not None:
            obj.pos = [-8.0, 3.0 if name == "object1" else -3.0, 0.1]
    add_bags(spec, card_dir)
    vis = getattr(spec.visual, "global")                                      # room for big camera images
    vis.offwidth, vis.offheight = max(vis.offwidth, 1920), max(vis.offheight, 1080)
    for i, p in enumerate(people):
        photo = find_photo(p["name"], p.get("photo"))
        card = photo_card(os.path.join(card_dir, f"person_{i}.png"), photo) if photo else None
        png, card_w, card_h = card or drawn_card(os.path.join(card_dir, f"person_{i}.png"), i)
        spec.add_texture(name=f"tex_person_{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
        mat = spec.add_material(name=f"mat_person_{i}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tex_person_{i}"
        x, y = p["pos"]
        w, h = card_w / 2, card_h / 2
        # The picture: a flat textured slab facing +x (the side we turn toward the robot).
        spec.add_mesh(name=f"mesh_person_{i}",
                      uservert=[0.006, -w, -h, 0.006, w, -h, 0.006, w, h, 0.006, -w, h,     # front
                                0.002, -w, -h, 0.002, w, -h, 0.002, w, h, 0.002, -w, h],    # back: gives it volume
                      userface=[0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,       # closed slab
                                3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2],
                      usertexcoord=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        body = spec.worldbody.add_body(name=f"person_{i}", pos=[x, y, h], mocap=True)
        body.add_geom(name=f"person_{i}_picture", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh_person_{i}",
                      material=f"mat_person_{i}", contype=0, conaffinity=0, group=1)
        # A thin box behind it, so the lidar sees people (like legs) and the robot can't drive
        # through them. Group 3 is hidden in the camera images.
        body.add_geom(name=f"person_{i}_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, w, h],
                      rgba=[0.6, 0.55, 0.45, 1], group=3)
    return spec.compile()


# ---- the swag table, goodie bags in lanes, lane markers -------------------------------------
TABLE_CENTER = (-1.0, 0.0)             # stock table, turned: 1.2 m along y, 1.0 m deep, 0.48 m tall
TABLE_TOP = 0.48                       # -> front edge at x = -0.5, 0.5 m behind the start spot
TABLE_EDGE_X = TABLE_CENTER[0] + 0.5
BAG_W, BAG_D, BAG_H = 0.16, 0.10, 0.22  # width (along the table edge), depth, height
HANDLE_H, HANDLE_R = 0.08, 0.005       # loops rise this far above the bag; rope radius
# Each item's bags have their own colour (bag_detect checks it) and their own lane: a row running
# from the table's front edge to the back, one bag behind the other. Seen from in front of the
# table: pens on the left, t-shirts in the middle, plushies on the right.
BAG_COLOURS = {"pen": (205, 30, 35), "shirt": (30, 70, 200), "plushie": (35, 160, 60)}   # red, blue, green
LANES = {"pen": -0.25, "shirt": 0.0, "plushie": 0.25}  # lane centre (y); the robot faces -x at the table
FIRST_SLOT_M, SLOT_PITCH_M = 0.08, 0.13  # table edge -> middle of the front bag; bag to bag in a lane
BAGS_PER_LANE = 2
# A marker (ArUco, DICT_6X6_250) on a post at the back of each lane, above the bags' loops, facing the
# open side: the robot finds the table and the lanes by these.
MARKER_IDS = {"pen": 20, "shirt": 21, "plushie": 22}
MARKER_SIZE_M = 0.10                   # the black square
MARKER_TO_EDGE_M = 0.45                # marker face -> table front edge
MARKER_Z = 1.0                         # middle of the marker above the floor
BAGS = [(item, y, TABLE_EDGE_X - FIRST_SLOT_M - k * SLOT_PITCH_M)
        for item, y in LANES.items() for k in range(BAGS_PER_LANE)]


def bag_front(path: str, item: str) -> str:
    """The bag's front: its colour, with the item's name in white (for people; the robot uses the colour)."""
    from PIL import Image, ImageDraw, ImageFont

    px_per_m = 2000
    w, h = int(BAG_W * px_per_m), int(BAG_H * px_per_m)
    img = Image.new("RGB", (w, h), BAG_COLOURS[item])
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((w // 2, int(0.06 * px_per_m)), item.upper(), fill=(255, 255, 255), font=font,
                             anchor="mt")
    img.save(path)
    return path


def marker_card(path: str, marker_id: int) -> tuple[str, float]:
    """The printed marker: the ArUco square with a white margin. Returns (png, card width in m)."""
    import cv2

    cell_px = 40                                                 # 6x6 + 1-cell black border = 8 cells
    square = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                                           marker_id, 8 * cell_px)
    margin = 2 * cell_px                                         # quiet zone: 2 cells of white
    card = cv2.copyMakeBorder(square, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, cv2.cvtColor(card, cv2.COLOR_GRAY2BGR))
    return path, MARKER_SIZE_M * card.shape[0] / square.shape[0]


def slab(spec, name: str, x: float, w: float, h: float):
    """A flat textured card facing +x (front at x, 1 mm thick), as a mesh with texture coordinates."""
    spec.add_mesh(name=name,
                  uservert=[x, -w, -h, x, w, -h, x, w, h, x, -w, h,
                            x - 0.001, -w, -h, x - 0.001, w, -h, x - 0.001, w, h, x - 0.001, -w, h],
                  userface=[0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
                            3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2],
                  usertexcoord=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def add_bags(spec, card_dir: str) -> None:
    """Paper goodie bags in their lanes, coloured by item: a free body with two white rope loops on
    top (front and back), the item's name on the front (+x, the open side). Plus the lane markers."""
    spec.add_material(name="rope", rgba=[0.97, 0.97, 0.97, 1])
    d, w, h = BAG_D / 2, BAG_W / 2, BAG_H / 2
    for item, rgb in BAG_COLOURS.items():
        spec.add_material(name=f"paper_{item}", rgba=[c / 255 for c in rgb] + [1])
        png = bag_front(os.path.join(card_dir, f"bag_{item}.png"), item)
        spec.add_texture(name=f"tex_bag_{item}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
        mat = spec.add_material(name=f"mat_bag_{item}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tex_bag_{item}"
    jitter = float(os.environ.get("SWAG_BAG_JITTER", "0"))     # test: bags put down sloppily (+-m, +-yaw)
    rng = np.random.default_rng(int(os.environ.get("SWAG_BAG_SEED", "1")))
    for k, (item, y, x0) in enumerate(BAGS):
        name = f"{item}_{k}"
        if jitter:
            x0, y = x0 + rng.uniform(-0.015, 0.015), y + rng.uniform(-jitter, jitter)   # along the lane: against a tape mark
        body = spec.worldbody.add_body(name=f"bag_{name}", pos=[x0, y, TABLE_TOP + h + 0.002])
        body.add_freejoint()
        body.add_geom(name=f"bag_{name}_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[d, w, h],
                      material=f"paper_{item}", mass=0.15, friction=[1.0, 0.01, 0.001])
        slab(spec, f"mesh_bag_{name}", d + 0.0015, w, h)          # the printed front, just in front of the +x face
        body.add_geom(name=f"bag_{name}_front", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh_bag_{name}",
                      material=f"mat_bag_{item}", contype=0, conaffinity=0, mass=0.0001, group=1)
        for side in (-1, 1):                                    # front and back loop
            lx, top, leg_y = side * (d - 0.012), h + HANDLE_H, 0.05
            for fromto in ([lx, -leg_y, h - 0.02, lx, -leg_y, top], [lx, leg_y, h - 0.02, lx, leg_y, top],
                           [lx, -leg_y, top, lx, leg_y, top]):
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=fromto, size=[HANDLE_R, 0, 0],
                              material="rope", mass=0.003, friction=[2.0, 0.05, 0.01], condim=6)
    spec.add_material(name="post", rgba=[0.35, 0.3, 0.25, 1])
    mx = TABLE_EDGE_X - MARKER_TO_EDGE_M
    for item, y in LANES.items():
        png, card_w = marker_card(os.path.join(card_dir, f"marker_{MARKER_IDS[item]}.png"), MARKER_IDS[item])
        spec.add_texture(name=f"tex_marker_{item}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png)
        mat = spec.add_material(name=f"mat_marker_{item}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tex_marker_{item}"
        body = spec.worldbody.add_body(name=f"marker_{item}", pos=[mx, y, MARKER_Z])
        slab(spec, f"mesh_marker_{item}", 0.0, card_w / 2, card_w / 2)
        body.add_geom(name=f"marker_{item}_card", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh_marker_{item}",
                      material=f"mat_marker_{item}", contype=0, conaffinity=0, group=1)
        body.add_geom(name=f"marker_{item}_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, material="post",
                      fromto=[-0.01, 0, TABLE_TOP - MARKER_Z, -0.01, 0, -card_w / 2], size=[0.008, 0, 0])


def face_toward(model, data, robot_xy) -> None:
    """Turn every cutout so its picture faces the robot."""
    for i in range(model.nmocap):
        px, py = data.mocap_pos[i][:2]
        yaw = math.atan2(robot_xy[1] - py, robot_xy[0] - px)
        data.mocap_quat[i] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


# ---- faster physics loop ----------------------------------------------------------------
SENSOR_BIT = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)


class FastLoopMixin:
    """Overrides MujocoServer's per-step loop (see module docstring)."""

    def __init__(self, scene_xml_path, model, *args, **kwargs):
        from stretch_mujoco import utils
        if model is None:
            people = load_world(os.environ.get("SWAG_WORLD") or None)
            model = build_model(scene_xml_path or utils.default_scene_xml_path, people,
                                os.environ.get("SWAG_CARD_DIR", "/tmp/stretch_swag_cards"))
            print(f"[swag] {len(people)} people in the scene: " +
                  ", ".join(f"{p['name']} ({'photo' if find_photo(p['name'], p.get('photo')) else 'drawn'})" for p in people),
                  flush=True)
        super().__init__(scene_xml_path, model, *args, **kwargs)
        self.speed = max(0.1, float(os.environ.get("SWAG_SIM_SPEED", "1.0")))
        self.people_file = os.environ.get("SWAG_PEOPLE_FILE", "")
        self.people_mtime = 0.0
        # Start with the head looking straight ahead at chest height (pan 0, tilt slightly down).
        for name, value in (("head_pan", 0.0), ("head_tilt", -0.1)):
            try:
                self.mjdata.actuator(name).ctrl = value
            except KeyError:
                pass

    def _compute_sensors(self) -> None:
        m, d = self.mjmodel, self.mjdata
        m.opt.disableflags = m.opt.disableflags & ~SENSOR_BIT
        mujoco.mj_sensorPos(m, d)
        mujoco.mj_sensorVel(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        mujoco.mj_sensorAcc(m, d)
        m.opt.disableflags = m.opt.disableflags | SENSOR_BIT

    def _read_people(self) -> None:
        """Positions written by sim_people (people who walk away move here too)."""
        if not self.people_file:
            return
        try:
            mtime = os.path.getmtime(self.people_file)
            if mtime == self.people_mtime:
                return
            self.people_mtime = mtime
            positions = json.loads(open(self.people_file).read())["positions"]
        except (OSError, ValueError, KeyError):
            return
        for i, (x, y) in enumerate(positions[: self.mjmodel.nmocap]):
            self.mjdata.mocap_pos[i][:2] = (x, y)

    def _physics_loop(self, lock, termination_check):
        m, d = self.mjmodel, self.mjdata
        m.opt.disableflags = m.opt.disableflags | SENSOR_BIT
        chunk = max(1, round(0.02 / m.opt.timestep))             # sync with the driver at 50 Hz (sim time)
        print(f"[swag] fast physics: {self.speed:g}x real time, lidar 15 Hz, status sync 50 Hz", flush=True)
        wall0, sim0 = time.perf_counter(), d.time
        last_sense = last_people = last_report = last_bags = 0.0
        bag_ids = [i for i in range(m.nbody) if m.body(i).name.startswith("bag_")]
        bag_seen = {i: d.xpos[i].copy() for i in bag_ids}
        while termination_check():
            with lock:
                for _ in range(chunk):
                    mujoco.mj_step(m, d)
                self._ctrl_callback(m, d)
                now = time.perf_counter()
                if now - last_sense >= 1 / 15:
                    self._compute_sensors()
                    last_sense = now
                if now - last_people >= 0.1:
                    self._read_people()
                    face_toward(m, d, d.body("base_link").xpos[:2])
                    last_people = now
            if now - last_bags > 2.0:                               # report bags that moved (picked, dropped)
                last_bags = now
                for i in bag_ids:
                    p = d.xpos[i]
                    if np.linalg.norm(p - bag_seen[i]) > 0.05:
                        bag_seen[i] = p.copy()
                        print(f"[swag] {m.body(i).name} now at ({p[0]:.2f}, {p[1]:.2f}), {p[2]:.2f} m up", flush=True)
            ahead = (d.time - sim0) / self.speed - (time.perf_counter() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.5:                                     # can't keep up: don't try to catch up
                wall0, sim0 = time.perf_counter(), d.time
            if now - last_report > 30:
                rate = (d.time - sim0) / max(1e-6, time.perf_counter() - wall0)
                print(f"[swag] sim running at {rate:.2f}x real time", flush=True)
                last_report = now


def server_classes():
    from stretch_mujoco.mujoco_server import MujocoServer
    from stretch_mujoco.mujoco_server_passive import MujocoServerPassive

    class FastServer(FastLoopMixin, MujocoServer):
        pass

    class FastPassiveServer(FastLoopMixin, MujocoServerPassive):
        pass

    return FastServer, FastPassiveServer


patch_head_camera_resolution()

# Module-level names so the spawned MuJoCo process can unpickle `launch_server`.
FastServer, FastPassiveServer = server_classes()
FastServer.__qualname__ = FastServer.__name__ = "FastServer"
FastPassiveServer.__qualname__ = FastPassiveServer.__name__ = "FastPassiveServer"
FastServer.__module__ = FastPassiveServer.__module__ = __name__
