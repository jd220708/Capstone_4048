#!/usr/bin/env python3
"""
Estimate the sphere's world position and velocity from the drone's camera,
and log it alongside Gazebo's ground truth.

This is the measurement layer. The drone-following controller reads from it,
so get this right before letting the drone fly itself anywhere.

Run (sim, bridge, detector and gimbal tracker already up):
    rosenv
    python3 target_estimator.py

Writes ~/target_log_<timestamp>.csv and publishes /target_estimator/target.


HOW THE ESTIMATE WORKS
----------------------
The sphere always travels at z = 1.0, so the problem reduces to intersecting
the camera ray with a known horizontal plane:

  1. The detection's offset from frame centre gives an angle within the
     camera's field of view.
  2. Adding the gimbal angles and the drone's heading puts that ray into
     world coordinates.
  3. The drone's height above the sphere's plane is known, so

         horizontal range = (drone_z - 1.0) / tan(depression)

  4. Range plus bearing gives the sphere's x, y.
  5. Velocity is the position difference over time, smoothed.

Far more stable than estimating range from apparent size, which gets noisy
exactly where it matters - at long range, where the sphere is a few pixels.


CHECKING IT
-----------
The CSV carries both the estimate and Gazebo's true values, plus the error
between them. Fly for a minute, then look at err_dist:

  - small and steady          -> working
  - mirrored (x or y flipped) -> try --invert-gimbal-yaw
  - rotated by 180 degrees    -> try --gimbal-yaw-offset 3.14159
  - grows with range          -> check --hfov matches the camera

The gimbal is mounted rotated 180 degrees in yaw (x500_gimbal/model.sdf), so
a yaw offset or inversion is likely. The error columns tell you which.
"""

import argparse
import csv
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SPHERE_Z = 1.0          # the plane the sphere travels on
MAX_TRUE_SPEED = 50.0   # m/s; faster than this means a respawn, not motion
# The pink route sphere is named sphere_pink_<n> by the plugin. The decoy
# traffic is sphere_group_<route>_<n>. Matching the wrong prefix compares the
# estimate against whichever decoy last moved, which makes every error and
# velocity column meaningless.
MODEL_PREFIX = "sphere_pink"
DRONE_NAME = "x500_gimbal_0"


FULL_COLUMNS = [
    "t", "drone_x", "drone_y", "drone_z",
    "drone_yaw_deg", "drone_pitch_deg", "drone_roll_deg",
    "gimbal_pitch_deg", "gimbal_yaw_deg",
    "det_cx", "det_cy", "det_conf",
    "est_x", "est_y", "est_range", "est_vx", "est_vy", "est_speed",
    "true_x", "true_y", "true_vx", "true_vy", "true_speed",
    "err_x", "err_y", "err_dist",
]

# Minimal set for downstream filtering: a measurement, the truth to check it
# against, the time base, and the camera angle the measurement came from.
# Everything a Kalman filter needs to be driven and validated.
SLIM_COLUMNS = [
    "t",
    "gimbal_pitch_deg",
    "est_x", "est_y",
    "est_vx", "est_vy",
    "true_x", "true_y",
]


# --- Gazebo ground truth (optional) ----------------------------------------
# The keyboard_offboard script uses the same API. If these imports fail,
# copy the import lines from that file - the version suffix may differ.
GZ_OK = True
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_v_pb2 import Pose_V
except ImportError:
    try:
        from gz.transport14 import Node as GzNode
        from gz.msgs11.pose_v_pb2 import Pose_V
    except ImportError:
        GZ_OK = False


def rot_pitch(d, a):
    """Rotate about the body y axis. Negative a tilts the ray downwards."""
    c, s_ = math.cos(a), math.sin(a)
    return (c * d[0] - s_ * d[2], d[1], s_ * d[0] + c * d[2])


def rot_yaw(d, a):
    c, s_ = math.cos(a), math.sin(a)
    return (c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], d[2])


def rot_quat(d, q):
    """Rotate a vector by a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z)) * d[0] + 2 * (x * y - z * w) * d[1] + 2 * (x * z + y * w) * d[2],
        2 * (x * y + z * w) * d[0] + (1 - 2 * (x * x + z * z)) * d[1] + 2 * (y * z - x * w) * d[2],
        2 * (x * z - y * w) * d[0] + 2 * (y * z + x * w) * d[1] + (1 - 2 * (x * x + y * y)) * d[2],
    )


def quat_rpy(q):
    """Roll, pitch, yaw from a quaternion (x, y, z, w)."""
    x, y, z, w = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def quat_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TargetEstimator(Node):
    def __init__(self, a):
        super().__init__("target_estimator")
        self.a = a

        # latest inputs
        self.det = None            # (cx, cy, conf)
        self.det_t = 0.0
        self.gim = (0.0, 0.0)      # pitch, yaw in radians
        self.drone = None          # (x, y, z, yaw)
        self.drone_q = (0.0, 0.0, 0.0, 1.0)   # full attitude
        self.truth = None          # (x, y, z)
        self.truth_prev = None
        self.truth_prev_t = None
        self.true_vel = (0.0, 0.0)

        # velocity estimation
        self.hist = deque(maxlen=a.vel_window)
        self.smooth_xy = None
        self.last_fix = 0.0

        self.create_subscription(String, "/sphere_detector/detections",
                                 self.on_det, 10)
        self.create_subscription(String, "/gimbal_tracker/status",
                                 self.on_gimbal, 10)
        self.pub = self.create_publisher(String, "/target_estimator/target", 10)

        # Gazebo pose feed: drone pose, orientation and sphere ground truth
        self.gz = None
        if GZ_OK:
            self.gz = GzNode()
            topic = f"/world/{a.world}/pose/info"
            if self.gz.subscribe(Pose_V, topic, self.on_poses):
                self.get_logger().info(f"Gazebo poses: {topic}")
            else:
                self.get_logger().error(f"Could not subscribe to {topic}")
                self.gz = None
        else:
            self.get_logger().error(
                "gz transport Python bindings not found - no drone pose and no "
                "ground truth. Check the import lines in keyboard_offboard.py.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(a.out or Path.home() / f"target_log_{stamp}.csv")
        self.fh = open(self.path, "w", newline="")
        self.csv = csv.writer(self.fh)
        self.columns = SLIM_COLUMNS if a.slim else FULL_COLUMNS
        self.csv.writerow(self.columns)

        self.t0 = time.monotonic()
        self.rows = 0
        self.create_timer(1.0 / a.rate, self.tick)

        self.get_logger().info(f"Logging to {self.path}")
        self.get_logger().info(f"HFOV {math.degrees(a.hfov):.1f} deg, "
                               f"sphere plane z={SPHERE_Z}")

    # ------------------------------------------------------------------
    def on_det(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        dets = d.get("detections", [])
        if not dets:
            self.det = None
            return
        best = max(dets, key=lambda x: x["conf"])
        self.det = (best["centre_norm"][0], best["centre_norm"][1], best["conf"])
        self.det_t = time.monotonic()

    def on_gimbal(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.gim = (math.radians(d.get("pitch_deg", 0.0)),
                    math.radians(d.get("yaw_deg", 0.0)))

    def on_poses(self, msg):
        """Gazebo pose feed - drone state and sphere ground truth."""
        now = time.monotonic()
        for p in msg.pose:
            if p.name == DRONE_NAME:
                q = p.orientation
                self.drone = (p.position.x, p.position.y, p.position.z,
                              quat_yaw(q.x, q.y, q.z, q.w))
                self.drone_q = (q.x, q.y, q.z, q.w)
            elif p.name.startswith(self.a.target_prefix):
                new = (p.position.x, p.position.y, p.position.z)
                if self.truth_prev is not None and self.truth_prev_t is not None:
                    dt = now - self.truth_prev_t
                    jump = math.hypot(new[0] - self.truth_prev[0],
                                      new[1] - self.truth_prev[1])
                    # One sphere finishing its route and another spawning
                    # elsewhere looks like a single object teleporting. Anything
                    # faster than MAX_TRUE_SPEED is a respawn, not motion.
                    if dt > 1e-3 and jump / dt < MAX_TRUE_SPEED:
                        self.true_vel = ((new[0] - self.truth_prev[0]) / dt,
                                         (new[1] - self.truth_prev[1]) / dt)
                    elif dt > 1e-3:
                        # Target respawned. Reset the ground-truth velocity,
                        # but leave the estimate's history alone - the two are
                        # independent, and clearing it here starved the
                        # velocity fit of samples.
                        self.true_vel = (0.0, 0.0)
                self.truth_prev, self.truth_prev_t = new, now
                self.truth = new

    # ------------------------------------------------------------------
    def estimate(self):
        """Camera ray, rotated by the full drone attitude, hit against z = SPHERE_Z.

        The earlier version used only the drone's yaw, which is fine while
        hovering but wrong the moment it flies: a multirotor accelerating at
        10 m/s banks 20-30 degrees, and that tilt swings the camera. At 75 m
        altitude a 20 degree error moves the computed ground point by
        hundreds of metres. Using the full quaternion removes that entirely.
        """
        if self.det is None or self.drone is None:
            return None
        if time.monotonic() - self.det_t > 0.5:
            return None                     # stale detection

        cx, cy, _ = self.det
        dx, dy, dz, _ = self.drone
        gpitch, gyaw = self.gim

        # The detection and the gimbal angles are sampled independently, so a
        # frame is paired with an angle measured a moment later. While the
        # gimbal is slewing that mismatch is large, and it shows up as error
        # proportional to how far off-centre the target is. Measured on real
        # flight data: 7 m error within 2.6% of centre, 34 m beyond 7%.
        # Discarding off-centre frames costs data rate and buys accuracy.
        if math.hypot(cx - 0.5, cy - 0.5) > self.a.max_offset:
            return None

        if self.a.invert_gimbal_yaw:
            gyaw = -gyaw
        gyaw += self.a.gimbal_yaw_offset

        vfov = 2 * math.atan(math.tan(self.a.hfov / 2) / self.a.aspect)
        az = (cx - 0.5) * self.a.hfov        # + : target right of centre
        el = -(cy - 0.5) * vfov              # + : target above centre

        # Ray in the camera's own frame: x forward, z up.
        d = (math.cos(el) * math.cos(az),
             math.cos(el) * math.sin(az),
             math.sin(el))
        d = rot_pitch(d, gpitch)             # gimbal tilt
        d = rot_yaw(d, gyaw)                 # gimbal pan  -> body frame
        d = rot_quat(d, self.drone_q)        # drone attitude -> world frame

        if d[2] > -0.02:                     # not looking down
            return None

        t = (SPHERE_Z - dz) / d[2]           # distance along the ray
        if t <= 0:
            return None

        ex, ey = dx + t * d[0], dy + t * d[1]
        rng = math.hypot(ex - dx, ey - dy)
        if rng <= 0.0 or rng > self.a.max_range:
            return None
        return (ex, ey, rng)

    def velocity(self, x, y, t):
        """Least-squares slope over a short window - steadier than differencing.

        Clears its history when the estimate jumps. A jump means the detector
        has switched to a different sphere - one route ending, the next
        spawning - and fitting a line across that discontinuity yields a wild
        velocity. Which then gets multiplied by the lag compensation.
        """
        if self.hist:
            lt, lx, ly = self.hist[-1]
            dt = t - lt
            if dt > 1e-3 and math.hypot(x - lx, y - ly) / dt > self.a.max_speed:
                self.hist.clear()
        self.hist.append((t, x, y))
        if len(self.hist) < 3:
            return 0.0, 0.0
        ts = [h[0] for h in self.hist]
        tm = sum(ts) / len(ts)
        den = sum((tt - tm) ** 2 for tt in ts)
        if den < 1e-6:
            return 0.0, 0.0
        xm = sum(h[1] for h in self.hist) / len(self.hist)
        ym = sum(h[2] for h in self.hist) / len(self.hist)
        vx = sum((h[0] - tm) * (h[1] - xm) for h in self.hist) / den
        vy = sum((h[0] - tm) * (h[2] - ym) for h in self.hist) / den
        # Final guard: never report a speed the target can't physically reach.
        if math.hypot(vx, vy) > self.a.max_speed:
            return 0.0, 0.0
        return vx, vy

    # ------------------------------------------------------------------
    def tick(self):
        t = time.monotonic() - self.t0
        est = self.estimate()

        if est is None:
            # No usable measurement this tick - the target drifted off centre,
            # the detection went stale, or the geometry was unusable. That is
            # NOT a reason to forget the target's velocity: the last estimate
            # is still the best available. Clearing here reset the velocity to
            # zero on 19% of ticks and was the dominant source of the 0-to-36
            # m/s swings the follower was acting on.
            #
            # Only a long outage means the history is genuinely stale.
            if self.hist and time.monotonic() - self.last_fix > self.a.hist_timeout:
                self.hist.clear()
                self.smooth_xy = None
            return

        self.last_fix = time.monotonic()

        ex, ey, rng = est

        # Smooth before differentiating. Raw estimates jitter by tens of
        # metres between samples, and differentiating that produces nonsense
        # velocities which then trip the guard below, leaving no velocity at
        # all. Smoothing first keeps the velocity fit fed.
        if self.smooth_xy is None:
            self.smooth_xy = (ex, ey)
        else:
            a_ = self.a.smooth
            self.smooth_xy = (a_ * ex + (1 - a_) * self.smooth_xy[0],
                              a_ * ey + (1 - a_) * self.smooth_xy[1])
        ex, ey = self.smooth_xy

        vx, vy = self.velocity(ex, ey, t)
        speed = math.hypot(vx, vy)

        # The pipeline - capture, inference, gimbal settling - takes time, so
        # the estimate describes where the target WAS. Project it forward by
        # the measured lag. Measure your own with the CSV before setting this;
        # leave at 0 to log the raw estimate.
        if self.a.lag > 0.0:
            ex += vx * self.a.lag
            ey += vy * self.a.lag

        tx = ty = tvx = tvy = tspeed = ""
        errx = erry = errd = ""
        if self.truth is not None:
            tx, ty = self.truth[0], self.truth[1]
            tvx, tvy = self.true_vel
            tspeed = math.hypot(tvx, tvy)
            errx, erry = ex - tx, ey - ty
            errd = math.hypot(errx, erry)

        dx, dy, dz, dyaw = self.drone
        droll, dpitch, _ = quat_rpy(self.drone_q)
        cx, cy, conf = self.det

        def num(v, places=2):
            return "" if v == "" else f"{v:.{places}f}"

        row = {
            "t":                f"{t:.3f}",
            "drone_x":          num(dx),
            "drone_y":          num(dy),
            "drone_z":          num(dz),
            "drone_yaw_deg":    num(math.degrees(dyaw)),
            "drone_pitch_deg":  num(math.degrees(dpitch)),
            "drone_roll_deg":   num(math.degrees(droll)),
            "gimbal_pitch_deg": num(math.degrees(self.gim[0])),
            "gimbal_yaw_deg":   num(math.degrees(self.gim[1])),
            "det_cx":           num(cx, 4),
            "det_cy":           num(cy, 4),
            "det_conf":         num(conf, 3),
            "est_x":            num(ex),
            "est_y":            num(ey),
            "est_range":        num(rng),
            "est_vx":           num(vx),
            "est_vy":           num(vy),
            "est_speed":        num(speed),
            "true_x":           num(tx),
            "true_y":           num(ty),
            "true_vx":          num(tvx),
            "true_vy":          num(tvy),
            "true_speed":       num(tspeed),
            "err_x":            num(errx),
            "err_y":            num(erry),
            "err_dist":         num(errd),
        }

        self.csv.writerow([row[c] for c in self.columns])
        self.rows += 1
        if self.rows % 50 == 0:
            self.fh.flush()

        self.pub.publish(String(data=json.dumps({
            "t": t, "x": round(ex, 2), "y": round(ey, 2), "z": SPHERE_Z,
            "range": round(rng, 2),
            "vx": round(vx, 2), "vy": round(vy, 2), "speed": round(speed, 2),
        })))

        if self.rows % (self.a.rate * 2) == 0:
            if errd != "":
                self.get_logger().info(
                    f"est ({ex:7.1f},{ey:7.1f})  true ({tx:7.1f},{ty:7.1f})  "
                    f"err {errd:6.1f} m   speed est {speed:4.1f} true {tspeed:4.1f}")
            else:
                self.get_logger().info(
                    f"est ({ex:7.1f},{ey:7.1f})  range {rng:6.1f}  speed {speed:4.1f}")

    def close(self):
        self.fh.flush()
        self.fh.close()
        print(f"\n{self.rows} rows written to {self.path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="city_test")
    p.add_argument("--target-prefix", default=MODEL_PREFIX,
                   help="Model-name prefix of the tracked target in Gazebo.")
    p.add_argument("--hfov", type=float, default=1.047,
                   help="Camera horizontal FOV, radians. Must match the model.")
    p.add_argument("--aspect", type=float, default=16 / 9)
    p.add_argument("--rate", type=int, default=10, help="Log rate, Hz.")
    p.add_argument("--hist-timeout", type=float, default=2.0,
                   help="Seconds without a usable measurement before the velocity "
                        "history is discarded as stale.")
    p.add_argument("--vel-window", type=int, default=25,
                   help="Samples used for the velocity fit.")
    p.add_argument("--max-range", type=float, default=2000.0)
    p.add_argument("--max-offset", type=float, default=0.05,
                   help="Discard detections further than this fraction of the "
                        "frame from centre - the gimbal is slewing and its "
                        "reported angle is stale. Lower is more accurate but "
                        "keeps fewer frames.")
    p.add_argument("--smooth", type=float, default=0.3,
                   help="Position smoothing, 0-1. Lower is smoother and laggier.")
    p.add_argument("--max-speed", type=float, default=25.0,
                   help="Plausible ceiling on target speed, m/s. Estimates above "
                        "this are discarded as target switches, not motion.")
    p.add_argument("--invert-gimbal-yaw", action="store_true")
    p.add_argument("--gimbal-yaw-offset", type=float, default=0.0,
                   help="Radians. Try 3.14159 if the estimate is 180 deg out.")
    p.add_argument("--lag", type=float, default=0.0,
                   help="Seconds of pipeline lag to compensate for. "
                        "Measure it from the CSV first; ~0.8 for this rig at 75 m.")
    p.add_argument("--slim", action="store_true",
                   help="Log only t, gimbal pitch, estimated and true position "
                        "and velocity - the columns a downstream filter needs.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rclpy.init()
    node = TargetEstimator(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
