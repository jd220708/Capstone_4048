#!/usr/bin/env python3
"""
Capture frames from the drone's camera for training data.

Subscribes to the bridged camera topic and saves frames at a fixed interval.

Usage:
    rosenv
    python3 capture_frames.py --session flight1

Run one session per flight, with a different --session name each time. The
session name becomes the filename prefix, which is what lets the training
script split train/val by flight instead of randomly. That split is the
difference between a validation score you can defend and one you can't.

Press Ctrl+C to stop. Frames land in ~/sphere_data/images/.

Deliberately does not use cv_bridge — it's compiled against NumPy 1.x and
breaks in this environment. The conversion is three lines anyway.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found. Run: pip install opencv-python --break-system-packages")


class FrameCapture(Node):
    def __init__(self, topic, out_dir, session, interval, limit):
        super().__init__("frame_capture")

        self.out_dir = Path(out_dir) / "images"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.session = session
        self.interval = interval
        self.limit = limit
        self.count = 0
        self.last_save = 0.0
        self.warned_encoding = False

        # Sensor data QoS - matches what ros_gz_bridge publishes.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, topic, self.on_image, qos)

        self.get_logger().info(f"Listening on {topic}")
        self.get_logger().info(f"Saving to {self.out_dir} every {interval}s")
        self.get_logger().info(f"Session prefix: {session}")
        if limit:
            self.get_logger().info(f"Will stop after {limit} frames")
        self.get_logger().info("Fly around. Ctrl+C when done.")

    def on_image(self, msg: Image):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_save < self.interval:
            return
        self.last_save = now

        img = self.to_bgr(msg)
        if img is None:
            return

        name = f"{self.session}_{self.count:05d}.png"
        cv2.imwrite(str(self.out_dir / name), img)
        self.count += 1

        if self.count % 10 == 0:
            self.get_logger().info(f"{self.count} frames saved")

        if self.limit and self.count >= self.limit:
            self.get_logger().info(f"Reached {self.limit} frames. Stopping.")
            raise SystemExit(0)

    def to_bgr(self, msg: Image):
        """ROS Image -> BGR numpy array, without cv_bridge."""
        enc = msg.encoding.lower()
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("rgb8", "bgr8"):
                img = buf.reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img
            if enc in ("rgba8", "bgra8"):
                img = buf.reshape(msg.height, msg.width, 4)
                code = cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR
                return cv2.cvtColor(img, code)
            if enc in ("mono8", "8uc1"):
                img = buf.reshape(msg.height, msg.width)
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        except ValueError as e:
            self.get_logger().error(f"Could not reshape image: {e}")
            return None

        if not self.warned_encoding:
            self.get_logger().error(f"Unhandled encoding '{msg.encoding}'")
            self.warned_encoding = True
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True,
                   help="Prefix for this flight, e.g. flight1. Use a new one each flight.")
    p.add_argument("--topic", default="/x500/camera/image")
    p.add_argument("--out", default=str(Path.home() / "sphere_data"))
    p.add_argument("--interval", type=float, default=0.5,
                   help="Seconds between saved frames (default 0.5)")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N frames (0 = unlimited)")
    args = p.parse_args()

    rclpy.init()
    node = FrameCapture(args.topic, args.out, args.session, args.interval, args.limit)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print(f"\n{node.count} frames saved to {node.out_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
