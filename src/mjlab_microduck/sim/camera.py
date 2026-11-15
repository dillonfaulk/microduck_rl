"""What a duck sees, in the format `mediad` captures.

MuJoCo renders RGB; `mediad` pins its pipeline to UYVY because that is what `v4l2src` can drive at
full rate off the rkisp — so the conversion happens here, on the side that knows it is a simulator.

**Frames do not go down the JSON link.** 640x360 UYVY is 460,800 bytes, and at 15 fps that is 6.9
MB/s — JSON would be absurd. So a camera is its own TCP port carrying length-prefixed raw frames:
four bytes of little-endian length, then the bytes, forever. No handshake, because there is nothing
to negotiate that both ends do not already have to agree on to be useful.

**Opt in, per duck.** Rendering is the most expensive thing in the simulator by a wide margin —
12.2 ms per 640x360 frame, measured, against 0.3 ms to step four ducks' physics. Four ducks with
cameras at 15 fps is most of a core; four without is nothing. Most sessions do not need one.
"""

from __future__ import annotations

import socket
import socketserver
import struct
import threading

import mujoco
import numpy as np

# 16:9 at a size the default offscreen framebuffer can hold — MuJoCo caps offscreen rendering at
# the model's `<global offwidth/offheight>`, which is 640x480 unless the scene says otherwise.
WIDTH = 640
HEIGHT = 360

# The sensor's rate is 30, but a rendered frame costs 12 ms and a duck that is being watched is
# usually being watched rather than raced. 15 halves the cost for something nobody can see.
FPS = 15

# BT.601, the same coefficients `duck_detect`'s one-pass sampler uses on the robot.
_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def to_uyvy(rgb: np.ndarray) -> bytes:
    """RGB to packed UYVY: `U Y0 V Y1` per pixel pair, chroma averaged across the pair.

    Averaged rather than dropped, because a subsampler that takes the left pixel's chroma puts a
    half-pixel colour shift into every frame — invisible on a duck and not invisible to a detector
    trained on a real camera.
    """
    frame = rgb.astype(np.float32)
    luma = frame @ _Y
    chroma_u = frame @ _U + 128.0
    chroma_v = frame @ _V + 128.0

    pairs = frame.shape[1] // 2
    packed = np.empty((frame.shape[0], pairs, 4), dtype=np.uint8)
    packed[:, :, 0] = np.clip((chroma_u[:, 0::2] + chroma_u[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 1] = np.clip(luma[:, 0::2], 0, 255)
    packed[:, :, 2] = np.clip((chroma_v[:, 0::2] + chroma_v[:, 1::2]) / 2.0, 0, 255)
    packed[:, :, 3] = np.clip(luma[:, 1::2], 0, 255)
    return packed.tobytes()


# The camera's axes in the ToF site's frame (site: +x forward, +y left, +z up; MuJoCo camera: looks
# along -z, +y is image up, +x image right). The site is on the same head body as the camera, so
# both turn with the head.
#   upright (rotate 0):        right = -y_site, up = +z_site, back = -x_site
#   quarter turn (rotate 90):  right = -z_site (world down, as the legacy twin), up = -y_site,
#                              back = -x_site. A consumer turns the frame 90 deg clockwise to
#                              upright, which is mediad's `rotate` 90.
_SITE_TO_CAM = {
    0: np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    90: np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]),
}


def apply_profile(model: mujoco.MjModel, camera: int, profile, ref_site: str = "tof") -> None:
    """Mount `camera` like the board's real camera: its rotation, and its field of view as MuJoCo's
    `fovy` (the unrotated frame's vertical). Render at the profile's aspect ratio (16:9 for both)
    to get its horizontal field too. Distortion is not rendered: `profile.camera.distortion` is
    what a detector should undo on real frames, and a rendered pinhole has none."""
    cam = profile.camera if hasattr(profile, "camera") else profile
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ref_site)
    if site < 0 or model.site_bodyid[site] != model.cam_bodyid[camera]:
        raise SystemExit(f"apply_profile needs a '{ref_site}' site on the camera's own body")
    if cam.rotate_deg not in _SITE_TO_CAM:
        raise SystemExit(f"no camera mount for rotate {cam.rotate_deg}")
    body_site = np.zeros(9)
    mujoco.mju_quat2Mat(body_site, model.site_quat[site])
    R = body_site.reshape(3, 3) @ _SITE_TO_CAM[cam.rotate_deg]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    model.cam_quat[camera] = q
    model.cam_fovy[camera] = cam.vfov_deg
    need_w, need_h = WIDTH * 2, HEIGHT * 2   # room for the profile's 1280x720 delivered size
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), need_w)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), need_h)


def pinhole_intrinsics(model: mujoco.MjModel, camera: int, width: int, height: int) -> dict:
    """fx, fy, cx, cy of a rendered frame (square pixels, centred principal point, no distortion):
    the numbers a detector should read instead of hard-coding a field of view."""
    fy = (height / 2.0) / np.tan(np.radians(model.cam_fovy[camera]) / 2.0)
    return {"fx": fy, "fy": fy, "cx": width / 2.0, "cy": height / 2.0, "distortion": []}


class Camera:
    """One duck's head camera, rendered on demand.

    The renderer is not thread-safe and is expensive to make, so one lives here and only the frame
    loop touches it.
    """

    def __init__(self, model: mujoco.MjModel, name: str, width: int = WIDTH, height: int = HEIGHT,
                 profile=None):
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.camera < 0:
            raise SystemExit(f"the model has no camera {name!r}")
        self.profile = profile
        if profile is not None:
            # A board profile (sim/board_profiles.py) sets the mount and the field of view from
            # the real camera's family solve. Without one, everything below is the twin as it
            # was: MuJoCo's default 45 deg fovy, on its side. `mediad --sim-camera` publishes
            # intrinsics for exactly that (mediad/src/camera.rs:53-55), so the legacy mode stays
            # the body server's default.
            apply_profile(model, self.camera, profile)
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.width, self.height = width, height
            self.latest: bytes | None = None
            self.lock = threading.Lock()
            return

        # **The model's head camera faces backwards.** Measured against the duck's own forward axis
        # and the ToF site's: the camera's view direction is -x where both of those are +x, exactly
        # 180 degrees out. On screen that is a duck apparently seeing what is behind it — a cyan cube
        # it is walking away from, sitting in frame.
        #
        # Turned 180 degrees about the camera's own **right** axis — not its up axis, which was the
        # first attempt and came out upside down. Both turns fix the direction; only this one leaves
        # the image the same way up. What the console has to undo is set by where the camera's right
        # axis points: the original camera has it along the world's *down*, and a yaw turn moves it
        # to *up*, so the quarter turn the console applies lands 180 degrees out.
        #
        # A roll turn keeps right pointing down, so a rendered frame comes out on its side exactly as
        # the original did — which is correct, because the real head camera is mounted a quarter turn
        # off and every consumer already expects that. `mediad --rotate 90` stays true of a simulated
        # duck for the same reason it is true of a real one.
        #
        # Done here rather than in the MJCF, because that file belongs to the RL work and a camera
        # nothing in training uses is not worth a change they have to review.
        turn = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degrees about x, scalar-first
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, model.cam_quat[self.camera], turn)
        model.cam_quat[self.camera] = fixed
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.width = width
        self.height = height
        self.latest: bytes | None = None
        self.lock = threading.Lock()

    def render(self, world) -> None:
        """Render one frame, reading `MjData` only while holding the world's lock.

        **`update_scene` reads the whole of `MjData`, and it runs on the step loop's thread while
        sensor reads run on socket threads.** Unlocked, a ToF read caught a site orientation
        mid-write and got a zero-length ray direction — which MuJoCo answers with
        `mj_ray: vector length is too small` and an abort, taking the simulator down with it. The
        lock is held for the scene copy, which is a millisecond, and released for the render, which
        is twelve and touches no shared state.
        """
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
        packed = to_uyvy(self.renderer.render())
        with self.lock:
            self.latest = packed

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest


class FrameHandler(socketserver.BaseRequestHandler):
    """Length-prefixed frames, at the camera's rate, until the reader goes away."""

    def handle(self) -> None:
        camera: Camera = self.server.camera
        fps: int = self.server.fps
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        period = 1.0 / max(1, fps)
        import time

        next_frame = time.perf_counter()
        try:
            while True:
                frame = camera.frame()
                if frame is not None:
                    self.request.sendall(struct.pack("<I", len(frame)) + frame)
                next_frame += period
                slack = next_frame - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_frame = time.perf_counter()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


class FrameServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
