#!/usr/bin/env python3
"""
Live pink-sphere detection and tracking on the drone's camera feed.

Subscribes to the bridged Gazebo camera, runs the trained YOLO model with
ByteTrack, and publishes:

  /sphere_detector/image_annotated   sensor_msgs/Image  - boxes, IDs, FPS
  /sphere_detector/detections        std_msgs/String    - JSON, one per frame

The JSON is deliberately simple so the loss-detection and reacquisition code
can consume it without extra dependencies. vision_msgs/Detection2DArray is the
"proper" ROS type for this and worth switching to later.

Run (sim and camera bridge already up):
    rosenv
    source ~/ros_yolo_env/bin/activate
    python3 sphere_detector.py

Watch it:
    rosenv
    ros2 run rqt_image_view rqt_image_view /sphere_detector/image_annotated

Order matters: source ROS first, then activate the venv.

Deliberately avoids cv_bridge, which is compiled against NumPy 1.x and breaks
in this environment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import cv2
except ImportError:
    sys.exit("OpenCV not found in this environment.")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not found. Did you activate ~/ros_yolo_env after sourcing ROS?")

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


DEFAULT_MODEL = Path.home() / "sphere_training/runs/sphere_n/weights/best.pt"


class SphereDetector(Node):
    def __init__(self, args):
        super().__init__("sphere_detector")

        if not Path(args.model).exists():
            self.get_logger().fatal(f"Model not found: {args.model}")
            raise SystemExit(1)

        self.model = YOLO(str(args.model))
        self.conf = args.conf
        self.imgsz = args.imgsz
        self.track = not args.no_track

        # Warm-up so the first real frame isn't slow and doesn't look like lag.
        self.model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8),
                           imgsz=args.imgsz, device=DEVICE, verbose=False)

        # Camera: best-effort, keep only the newest frame. If inference ever
        # falls behind, stale frames are dropped rather than queued up.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub = self.create_subscription(Image, args.topic, self.on_image, sensor_qos)

        # Outputs use default (reliable) QoS so rqt_image_view connects cleanly.
        self.pub_img = self.create_publisher(Image, "/sphere_detector/image_annotated", 5)
        self.pub_det = self.create_publisher(String, "/sphere_detector/detections", 10)

        self.frame_count = 0
        self.fps = 0.0
        self.t_last = time.monotonic()
        self.warned_encoding = False

        self.get_logger().info(f"Model:   {args.model}")
        self.get_logger().info(f"Device:  {'cuda:0' if DEVICE == 0 else 'cpu'}")
        self.get_logger().info(f"Input:   {args.topic}")
        self.get_logger().info(f"Mode:    {'tracking (ByteTrack)' if self.track else 'detection only'}")
        self.get_logger().info("Output:  /sphere_detector/image_annotated, /sphere_detector/detections")
        self.get_logger().info("Waiting for frames...")

    # ------------------------------------------------------------------
    def on_image(self, msg: Image):
        img = self.to_bgr(msg)
        if img is None:
            return

        t0 = time.monotonic()
        if self.track:
            results = self.model.track(img, persist=True, tracker="bytetrack.yaml",
                                       conf=self.conf, imgsz=self.imgsz,
                                       device=DEVICE, verbose=False)
        else:
            results = self.model.predict(img, conf=self.conf, imgsz=self.imgsz,
                                         device=DEVICE, verbose=False)
        infer_ms = (time.monotonic() - t0) * 1000
        r = results[0]

        detections = []
        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
            h, w = img.shape[:2]
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                detections.append({
                    "id": None if tid is None else int(tid),
                    "conf": round(float(c), 3),
                    "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                    # Centre as a fraction of the frame - what a gimbal
                    # controller wants: 0.5, 0.5 is dead centre.
                    "centre_norm": [round(float((x1 + x2) / 2 / w), 4),
                                    round(float((y1 + y2) / 2 / h), 4)],
                })

        self.pub_det.publish(String(data=json.dumps({
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "count": len(detections),
            "infer_ms": round(infer_ms, 1),
            "detections": detections,
        })))

        # FPS, smoothed over a one-second window
        self.frame_count += 1
        now = time.monotonic()
        if now - self.t_last >= 1.0:
            self.fps = self.frame_count / (now - self.t_last)
            self.frame_count = 0
            self.t_last = now

        annotated = r.plot()
        cv2.putText(annotated, f"{self.fps:4.1f} fps  {infer_ms:4.1f} ms  {len(detections)} tgt",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, f"{self.fps:4.1f} fps  {infer_ms:4.1f} ms  {len(detections)} tgt",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        self.pub_img.publish(self.to_msg(annotated, msg.header))

    # ------------------------------------------------------------------
    def to_bgr(self, msg: Image):
        """ROS Image -> BGR numpy array, without cv_bridge."""
        enc = msg.encoding.lower()
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("rgb8", "bgr8"):
                img = buf.reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img.copy()
            if enc in ("rgba8", "bgra8"):
                img = buf.reshape(msg.height, msg.width, 4)
                code = cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR
                return cv2.cvtColor(img, code)
        except ValueError as e:
            self.get_logger().error(f"Could not reshape image: {e}")
            return None
        if not self.warned_encoding:
            self.get_logger().error(f"Unhandled encoding '{msg.encoding}'")
            self.warned_encoding = True
        return None

    @staticmethod
    def to_msg(img: np.ndarray, header) -> Image:
        out = Image()
        out.header = header
        out.height, out.width = img.shape[:2]
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = img.shape[1] * 3
        out.data = np.ascontiguousarray(img).tobytes()
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--topic", default="/x500/camera/image")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--no-track", action="store_true", help="Plain detection, no IDs")
    args = p.parse_args()

    rclpy.init()
    node = SphereDetector(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
