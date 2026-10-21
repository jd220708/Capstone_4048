#!/usr/bin/env python3
"""
reacq_search_node.py

ROS 2 / Gazebo wrapper for the Bayesian reacquisition search (search_core.py).

While the target is tracked it only listens (the existing estimator, gimbal tracker and follower keep working).
When no detection has arrived for --t-lost seconds it FREEZES the Kalman filter, takes over the gimbal, runs the
search, and hands the gimbal back when the target is confirmed (or the search times out).

Reads:   /sphere_detector/detections   (JSON per frame)        /target_estimator/target (JSON x, y)
         /gimbal_tracker/status        (last gimbal angles)    Gazebo /world/<world>/pose/info (drone pose)
Writes:  gimbal pitch / yaw demand (Float64), camera zoom (Float64), /reacq/gimbal_sync, /reacq/state, CSV logs
Ground truth (--truth-prefix) is read ONLY to score results. It never reaches the search.

Run (sim, detector, gimbal_tracker, estimator, follower all up; ROS sourced, not in PX4's terminal):
    rosenv
    python3 reacq_search_node.py --strategy bayes
"""

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_v_pb2 import Pose_V
except ImportError:
    try:
        from gz.transport14 import Node as GzNode
        from gz.msgs11.pose_v_pb2 import Pose_V
    except ImportError:
        sys.exit("Gazebo Python transport bindings not found.")

from belief_grid import X_MIN, X_MAX, Y_MIN, Y_MAX, CELL_SIZE
from camera_model import Camera
from search_core import Config, SearchCore

STEP_COLS = ["t", "strategy", "state", "phase", "views", "aim_x", "aim_y", "pitch_deg", "yaw_deg",
             "peak_x", "peak_y", "peak_p", "entropy", "truth_x", "truth_y", "peak_to_truth_m"]
EP_COLS = ["strategy", "outcome", "t_loss", "duration_s", "views", "gimbal_deg", "plan_ms",
           "found_x", "found_y", "truth_x", "truth_y", "found_to_truth_m", "correct", "false_reacquisition"]


class ReacqSearch(Node):
    def __init__(self, a):
        super().__init__("reacq_search")
        self.a = a
        self.cfg = Config(bounds=tuple(a.bounds), cell_size=a.cell_size, t_lost=a.t_lost, max_search=a.max_search,
                          lam=a.lam, slew_rate=math.radians(a.slew_rate_deg), settle_s=a.settle,
                          meas_noise_std=a.meas_noise, ablate_prediction=a.no_prediction, ablate_miss=a.no_miss)
        self.cam = Camera(hfov=a.hfov, aspect=a.aspect, image_width_px=a.image_width, zoom=1.0,
                          yaw_offset=a.gimbal_yaw_offset, invert_yaw=a.invert_gimbal_yaw, target_z=a.target_z)
        self.core = SearchCore(self.cfg, self.cam, a.strategy, seed=a.seed)

        self.lock = threading.Lock()          # the Gazebo callback runs on its own thread
        self.sim_t = None                     # simulation time, from the pose message header
        self.drone = None                     # (position, quaternion)
        self.truth = None                     # scoring only
        self.last_det_t = None                # simulation time of the last accepted detection
        self.tracker_gimbal = None            # last (pitch, yaw) the gimbal tracker reported, radians

        self.create_subscription(String, a.estimate_topic, self.on_estimate, 10)
        self.create_subscription(String, a.detections_topic, self.on_detections, 10)
        self.create_subscription(String, a.status_topic, self.on_status, 10)
        self.pub_pitch = self.create_publisher(Float64, a.pitch_topic or f"/model/{a.drone_name}/command/gimbal_pitch_demo", 10)
        self.pub_yaw = self.create_publisher(Float64, a.yaw_topic or f"/model/{a.drone_name}/command/gimbal_yaw_demo", 10)
        self.pub_zoom = self.create_publisher(Float64, a.zoom_topic or f"/model/{a.drone_name}/camera/zoom/cmd_zoom", 10)
        self.pub_sync = self.create_publisher(String, "/reacq/gimbal_sync", 10)
        self.pub_state = self.create_publisher(String, "/reacq/state", 10)
        self.pub_follower_cmd = self.create_publisher(String, "/follower/cmd", 10)

        self.gz = GzNode()
        topic = f"/world/{a.world}/pose/info"
        if not self.gz.subscribe(Pose_V, topic, self.on_poses):
            self.get_logger().error(f"Could not subscribe to {topic}")

        os.makedirs(a.log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_file = open(os.path.join(a.log_dir, f"search_steps_{a.strategy}_{stamp}.csv"), "w", newline="")
        self.ep_file = open(os.path.join(a.log_dir, f"search_episodes_{a.strategy}_{stamp}.csv"), "w", newline="")
        self.step_w = csv.DictWriter(self.step_file, fieldnames=STEP_COLS)
        self.ep_w = csv.DictWriter(self.ep_file, fieldnames=EP_COLS)
        self.step_w.writeheader()
        self.ep_w.writeheader()
        self.tick = 0
        self.create_timer(1.0 / a.rate, self.on_timer)
        self.get_logger().info(f"Strategy {a.strategy}; loss declared after {a.t_lost}s without a detection")

    # ---------------------------------------------------------------- inputs
    def on_poses(self, msg):                                     # gz-transport thread
        t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        t = t if t > 0 else time.time()                          # fallback; only valid at real-time speed
        drone, truth = None, None
        for p in msg.pose:
            if p.name == self.a.drone_name or p.name.endswith(f"::{self.a.drone_name}"):
                q = p.orientation
                drone = ((p.position.x, p.position.y, p.position.z), (q.x, q.y, q.z, q.w))
            elif p.name.startswith(self.a.truth_prefix):
                truth = (p.position.x, p.position.y)
        with self.lock:
            self.sim_t = t
            if drone:
                self.drone = drone
            if truth:
                self.truth = truth

    def now(self):
        with self.lock:
            return self.sim_t

    def on_estimate(self, msg):
        t = self.now()
        if t is None:
            return
        try:
            d = json.loads(msg.data)
            # The estimate describes where the target WAS about meas_lag seconds ago, so stamp it that way.
            self.core.on_track_measurement(t - self.a.meas_lag, float(d["x"]), float(d["y"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def on_detections(self, msg):
        t = self.now()
        if t is None:
            return
        try:
            frame = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        dets = [{"cx": d["centre_norm"][0], "cy": d["centre_norm"][1], "conf": d["conf"]}
                for d in frame.get("detections", [])]
        if any(d["conf"] >= self.cfg.conf_min for d in dets):
            self.last_det_t = t
        self.core.on_detections(t, dets)

    def on_status(self, msg):
        try:
            d = json.loads(msg.data)
            p_rad = math.radians(d["pitch_deg"])
            y_rad = math.radians(d["yaw_deg"])
            self.tracker_gimbal = (p_rad, y_rad)
            if self.core.state == "SEARCH":
                self.core.gimbal.pitch = p_rad
                self.core.gimbal.yaw = y_rad
        except (json.JSONDecodeError, KeyError):
            pass

    # ---------------------------------------------------------------- main loop
    def on_timer(self):
        t = self.now()
        with self.lock:
            drone, truth = self.drone, self.truth
        if t is None or drone is None:
            return
        self.core.set_drone_pose(*drone)

        # ---- LOSS MONITOR: the rule that freezes the Kalman filter and starts the search ----
        # (gimbal_tracker.py only logs a loss; nothing else in the existing pipeline freezes a filter.)
        if self.core.state == "TRACK" and self.core.ready_for_loss() and self.last_det_t is not None:
            if t - self.last_det_t >= self.cfg.t_lost:
                self.begin_search(t)

        self.maybe_force_loss(t)

        out = self.core.tick(t)
        if out["cmd"] is not None:
            pitch, yaw = out["cmd"]
            self.pub_pitch.publish(Float64(data=float(pitch)))
            self.pub_yaw.publish(Float64(data=float(yaw)))
            self.sync(True, pitch, yaw)                          # keeps gimbal_tracker silent and in step
            self.log_step(t, truth)
        for name, te, info in out["events"]:
            self.get_logger().info(f"[{name}] {info}")
            if name in ("REACQUIRED", "TIMEOUT"):
                self.end_episode(truth)
        self.tick += 1
        if self.tick % 20 == 0:
            self.pub_state.publish(String(data=f"{self.core.state}/{getattr(self.core, 'phase', '-')}"))

    def maybe_force_loss(self, t):
        # FORCE-LOSS PLACEHOLDER ------------------------------------------------------------------
        # If you want the node itself to cause the loss for a trial, do it here, e.g. publish a gimbal command
        # that points the camera away from the target for N seconds, or stop passing detections on. Detections
        # then stop, and the loss monitor above declares the loss by itself. Losses forced by an external
        # script need nothing here. Do NOT read ground truth to decide when to force it.
        pass

    def begin_search(self, t):
        pitch, yaw = self.tracker_gimbal or (math.radians(-35.0), 0.0)
        if self.tracker_gimbal is None:
            self.get_logger().warn("No gimbal_tracker status seen; assuming pitch -35, yaw 0 as the start angle.")
        if self.core.declare_loss(t, pitch, yaw):
            self.pub_zoom.publish(Float64(data=1.0))             # the detection model assumes zoom 1.0
            self.sync(True, pitch, yaw)
            self.get_logger().warn("TARGET LOST - filter frozen, search started")

    def end_episode(self, truth):
        ep = self.core.last_episode
        self.sync(False, self.core.gimbal.pitch, self.core.gimbal.yaw)   # hand the gimbal back to the tracker
        
        # Automatically restart follower if successfully reacquired
        if ep["outcome"] == "REACQUIRED":
            self.pub_follower_cmd.publish(String(data="START"))
            self.get_logger().info("Target reacquired — restarting follower automatically.")

        row = {"strategy": ep["strategy"], "outcome": ep["outcome"], "t_loss": round(ep["t_loss"], 2),
               "duration_s": round(ep["duration"], 2), "views": ep["views"], "gimbal_deg": round(ep["gimbal_deg"], 1),
               "plan_ms": round(ep["compute_ms"], 1)}
        correct = False
        if ep["found_xy"] is not None:
            row.update(found_x=round(ep["found_xy"][0], 1), found_y=round(ep["found_xy"][1], 1))
            if truth is not None:                                # scoring only
                err = math.hypot(ep["found_xy"][0] - truth[0], ep["found_xy"][1] - truth[1])
                correct = err <= self.a.success_radius
                row.update(truth_x=round(truth[0], 1), truth_y=round(truth[1], 1), found_to_truth_m=round(err, 1))
        row.update(correct=correct, false_reacquisition=(ep["outcome"] == "REACQUIRED" and not correct))
        self.ep_w.writerow(row)
        self.ep_file.flush()
        self.get_logger().info(f"EPISODE: {row}")

    # ---------------------------------------------------------------- helpers
    def sync(self, hold, pitch, yaw):
        self.pub_sync.publish(String(data=json.dumps({"hold": hold, "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw)})))

    def log_step(self, t, truth):
        c = self.core
        px, py, pp = c.belief.peak()
        aim = c.aim or {}
        row = {"t": round(t, 3), "strategy": c.strategy, "state": c.state, "phase": c.phase, "views": c.views,
               "aim_x": round(aim.get("aim_xy", ("", ""))[0], 1) if aim else "",
               "aim_y": round(aim.get("aim_xy", ("", ""))[1], 1) if aim else "",
               "pitch_deg": round(math.degrees(c.gimbal.pitch), 2), "yaw_deg": round(math.degrees(c.gimbal.yaw), 2),
               "peak_x": round(px, 1), "peak_y": round(py, 1), "peak_p": round(pp, 4), "entropy": round(c.belief.entropy(), 3)}
        if truth is not None:                                    # scoring only
            row.update(truth_x=round(truth[0], 1), truth_y=round(truth[1], 1),
                       peak_to_truth_m=round(math.hypot(px - truth[0], py - truth[1]), 1))
        self.step_w.writerow(row)

    def destroy_node(self):
        self.step_file.close()
        self.ep_file.close()
        super().destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["bayes", "raster", "spiral", "predict"], default="bayes")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--world", default="city_test")
    p.add_argument("--drone-name", default="x500_gimbal_0")
    p.add_argument("--truth-prefix", default="sphere_pink", help="model-name prefix of the target, scoring only")
    p.add_argument("--bounds", type=float, nargs=4, default=[X_MIN, X_MAX, Y_MIN, Y_MAX], metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    p.add_argument("--cell-size", type=float, default=CELL_SIZE)
    p.add_argument("--detections-topic", default="/sphere_detector/detections")
    p.add_argument("--estimate-topic", default="/target_estimator/target")
    p.add_argument("--status-topic", default="/gimbal_tracker/status")
    p.add_argument("--pitch-topic", default="")
    p.add_argument("--yaw-topic", default="")
    p.add_argument("--zoom-topic", default="")
    # camera: copy the values you run target_estimator.py with
    p.add_argument("--hfov", type=float, default=1.047)
    p.add_argument("--aspect", type=float, default=16 / 9)
    p.add_argument("--image-width", type=int, default=1280)
    p.add_argument("--gimbal-yaw-offset", type=float, default=0.0)
    p.add_argument("--invert-gimbal-yaw", action="store_true")
    p.add_argument("--target-z", type=float, default=1.0)
    # search
    p.add_argument("--t-lost", type=float, default=2.0)
    p.add_argument("--max-search", type=float, default=90.0)
    p.add_argument("--lam", type=float, default=0.02)
    p.add_argument("--slew-rate-deg", type=float, default=60.0)
    p.add_argument("--settle", type=float, default=0.5)
    p.add_argument("--meas-lag", type=float, default=0.8, help="age of the estimator's output, seconds")
    p.add_argument("--meas-noise", type=float, default=8.0)
    p.add_argument("--success-radius", type=float, default=40.0)
    p.add_argument("--no-prediction", action="store_true", help="ablation")
    p.add_argument("--no-miss", action="store_true", help="ablation")
    p.add_argument("--rate", type=int, default=20)
    p.add_argument("--log-dir", default="reacq_logs")
    a = p.parse_args()

    rclpy.init()
    node = ReacqSearch(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
