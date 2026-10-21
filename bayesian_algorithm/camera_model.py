"""
camera_model.py

Camera geometry for the search, ported from target_estimator.py so the two always agree.

  forward model:  pixel -> camera ray -> gimbal pitch/yaw -> drone attitude -> world ray -> ground point
  inverse model:  ground point -> the gimbal pitch/yaw that puts it at the centre of the frame
  detection model: probability the detector finds a sphere standing at a ground point

Pure maths, no ROS. Works for any drone: pass its pose in, nothing is hard-coded to one vehicle.
"""

import math

import numpy as np


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quat_matrix(q):
    """Rotation matrix (body -> world) from a quaternion (x, y, z, w). Same maths as rot_quat() in the estimator."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_z(a):  # gimbal pan, same as rot_yaw() in the estimator
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_pitch(a):  # gimbal tilt, same as rot_pitch() in the estimator (negative a tilts the ray down)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


class Camera:
    def __init__(self, hfov=1.047, aspect=16 / 9, image_width_px=1280, zoom=1.0,
                 yaw_offset=0.0, invert_yaw=False, target_z=1.0, target_diameter=2.0,
                 p_max=0.9, s50_px=6.0, s_width_px=1.5, edge_margin=0.08):
        self.hfov = hfov                    # base horizontal field of view (rad) - MUST match the real camera
        self.aspect = aspect                # frame width / height
        self.image_width_px = image_width_px  # only used by the detection-size model
        self.zoom = zoom                    # 1.0 during search (the node publishes this)
        self.yaw_offset = yaw_offset        # same meaning as --gimbal-yaw-offset in target_estimator.py
        self.invert_yaw = invert_yaw        # same meaning as --invert-gimbal-yaw in target_estimator.py
        self.target_z = target_z            # height of the plane the target travels on
        self.target_diameter = target_diameter  # sphere diameter in metres (radius 1 in server.config)
        self.p_max = p_max                  # best-case detection probability (placeholder until measured)
        self.s50_px = s50_px                # apparent size (px) at which detection is 50% of p_max (placeholder)
        self.s_width_px = s_width_px        # softness of that transition (placeholder)
        self.edge_margin = edge_margin      # fraction of the frame over which detection fades near the edges

    # ---- field of view, including zoom ----
    @property
    def hfov_eff(self):
        return 2 * math.atan(math.tan(self.hfov / 2) / self.zoom)

    @property
    def vfov_eff(self):
        return 2 * math.atan(math.tan(self.hfov_eff / 2) / self.aspect)

    def _yaw_eff(self, yaw_cmd):
        """Commanded gimbal yaw -> effective body-frame yaw, applying the same correction as the estimator."""
        return (-yaw_cmd if self.invert_yaw else yaw_cmd) + self.yaw_offset

    def cam_to_world(self, q, pitch, yaw_cmd):
        """Rotation taking camera-frame vectors (x forward, y right, z up) into the world frame."""
        return quat_matrix(q) @ rot_z(self._yaw_eff(yaw_cmd)) @ rot_pitch(pitch)

    # ---- forward: pixel -> ground ----
    def pixel_to_ground(self, cx, cy, drone_pos, q, pitch, yaw_cmd):
        """Ground (x, y) seen at normalised pixel (cx, cy), or None if the ray never reaches the plane."""
        tan_h = math.tan(self.hfov_eff / 2)
        tan_v = math.tan(self.vfov_eff / 2)
        d_cam = np.array([1.0, (cx - 0.5) * 2 * tan_h, -(cy - 0.5) * 2 * tan_v])
        d = self.cam_to_world(q, pitch, yaw_cmd) @ d_cam
        if d[2] > -0.02:                        # not looking down: no usable intersection
            return None
        t = (self.target_z - drone_pos[2]) / d[2]
        if t <= 0:
            return None
        return float(drone_pos[0] + t * d[0]), float(drone_pos[1] + t * d[1])

    # ---- reverse: ground -> pixel ----
    def ground_to_pixel(self, pts, drone_pos, q, pitch, yaw_cmd):
        """For an (N, 2) array of ground points: normalised pixel x, y, slant range, and a 'in front of camera' mask."""
        P = np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), self.target_z)])
        v = P - np.asarray(drone_pos)[None, :]
        c = (self.cam_to_world(q, pitch, yaw_cmd).T @ v.T)      # points expressed in the camera frame
        front = c[0] > 1e-6
        cx_ = np.where(front, c[0], 1.0)
        tan_h = math.tan(self.hfov_eff / 2)
        tan_v = math.tan(self.vfov_eff / 2)
        px = 0.5 + (c[1] / cx_) / (2 * tan_h)
        py = 0.5 - (c[2] / cx_) / (2 * tan_v)
        return px, py, np.linalg.norm(v, axis=1), front

    # ---- detection probability P_D ----
    def detection_probability(self, pts, drone_pos, q, pitch, yaw_cmd):
        """
        Probability of detecting the target at each ground point for this camera pose.
        Zero outside the frame; fades near the frame edge (partly visible); falls with range because
        the sphere shrinks - a logistic in apparent size (pixels). All numbers are DOCUMENTED PLACEHOLDERS.
        """
        px, py, rng, front = self.ground_to_pixel(pts, drone_pos, q, pitch, yaw_cmd)
        inside = front & (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
        edge = np.minimum.reduce([px, 1 - px, py, 1 - py])
        f_edge = np.clip(edge / self.edge_margin, 0.0, 1.0)
        f_px = (self.image_width_px / 2) / math.tan(self.hfov_eff / 2)       # focal length in pixels
        size_px = f_px * self.target_diameter / np.maximum(rng, 1.0)         # apparent diameter in pixels
        f_size = 1.0 / (1.0 + np.exp(-(size_px - self.s50_px) / self.s_width_px))
        return np.where(inside, self.p_max * f_size * f_edge, 0.0)

    # ---- inverse: where to point ----
    def aim_angles(self, aim_xy, drone_pos, q, pitch_min=math.radians(-135), pitch_max=math.radians(45)):
        """Gimbal (pitch, yaw_cmd) that centres ground point aim_xy in the frame."""
        v = np.array([aim_xy[0] - drone_pos[0], aim_xy[1] - drone_pos[1], self.target_z - drone_pos[2]])
        b = quat_matrix(q).T @ (v / np.linalg.norm(v))          # direction in the drone body frame
        pitch = math.asin(clamp(b[2], -1.0, 1.0))
        yaw_eff = math.atan2(b[1], b[0])
        yaw_cmd = yaw_eff - self.yaw_offset
        if self.invert_yaw:
            yaw_cmd = -yaw_cmd
        return clamp(pitch, pitch_min, pitch_max), wrap_pi(yaw_cmd)


if __name__ == "__main__":
    # Self-test: aim at points, then check the forward model returns the same point (round trip).
    cam = Camera(yaw_offset=math.pi, invert_yaw=True)
    pos = (800.0, 700.0, 75.0)
    q = (0.0, 0.0, math.sin(0.4), math.cos(0.4))               # drone yawed 0.8 rad
    worst = 0.0
    for tx, ty in [(900, 760), (700, 650), (820, 900), (780, 705), (1000, 500)]:
        p, y = cam.aim_angles((tx, ty), pos, q)
        g = cam.pixel_to_ground(0.5, 0.5, pos, q, p, y)
        worst = max(worst, math.hypot(g[0] - tx, g[1] - ty))
        pd = cam.detection_probability(np.array([[tx, ty]]), pos, q, p, y)[0]
        print(f"aim ({tx},{ty}) -> pitch {math.degrees(p):6.1f} yaw {math.degrees(y):7.1f} | round trip ({g[0]:.1f},{g[1]:.1f}) | P_D at aim {pd:.2f}")
    print(f"worst round-trip error: {worst:.6f} m")
    print(f"effective FOV at zoom 1: {math.degrees(cam.hfov_eff):.1f} x {math.degrees(cam.vfov_eff):.1f} deg")