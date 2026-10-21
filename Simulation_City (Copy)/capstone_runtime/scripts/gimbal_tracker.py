#!/usr/bin/env python3
"""
Autonomous gimbal centring: lock onto one tracked sphere and keep it centred.

Reads /sphere_detector/detections, picks one target, and commands the gimbal
so that target sits at the middle of the frame. Proportional control with a
deadzone and a rate limit.

Requires the camera bridge to ALSO carry the gimbal topics - the camera-only
bridge command won't work. See the header note below.

Run (sim, detector and full bridge already up):
    rosenv
    python3 gimbal_tracker.py

Watch the error:
    rosenv
    ros2 topic echo /gimbal_tracker/status


HOW IT WORKS
------------
The detector reports each sphere's centre as a fraction of the frame, where
(0.5, 0.5) is dead centre. The offset from centre is converted to an angle
using the camera's field of view:

    yaw_error   = (cx - 0.5) * hfov
    pitch_error = (cy - 0.5) * vfov

then a fraction of that error (KP) is added to the current commanded angle,
limited to MAX_RATE per second. Small errors inside DEADZONE are ignored so
the gimbal doesn't chase detection noise.

This is a proportional controller. It will settle with a small steady-state
lag on a moving target - that's expected, and worth measuring rather than
immediately reaching for integral or derivative terms.


SIGN CONVENTIONS - READ THIS
----------------------------
From keyboard_offboard.py: UP arrow increases pitch ("camera up"), RIGHT
arrow increases yaw. But the gimbal is mounted rotated 180 degrees in yaw
(see x500_gimbal/model.sdf), so the yaw sign may be inverted relative to
what you'd expect.

If the gimbal swings AWAY from the target and runs to its limit, the sign is
wrong. Flip it with --invert-yaw or --invert-pitch. Test one axis at a time.
"""

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

# From keyboard_offboard.py - the physical travel of the gimbal
PITCH_MIN = math.radians(-135.0)
PITCH_MAX = math.radians(45.0)


class GimbalTracker(Node):
    def __init__(self, a):
        super().__init__("gimbal_tracker")

        self.a = a
        self.pitch = 0.0          # current commanded angle, radians
        self.yaw = 0.0
        self.locked_id = None     # track ID we're following
        self.last_seen = None     # monotonic time of last sighting
        self.last_state = None    # full state at last sighting, for loss logging
        self.declared_lost = False
        self.frames = 0

        self.sub = self.create_subscription(
            String, "/sphere_detector/detections", self.on_detections, 10)

        self.pub_pitch = self.create_publisher(
            Float64, "/model/x500_gimbal_0/command/gimbal_pitch_demo", 10)
        self.pub_yaw = self.create_publisher(
            Float64, "/model/x500_gimbal_0/command/gimbal_yaw_demo", 10)
        self.pub_status = self.create_publisher(String, "/gimbal_tracker/status", 10)
        ## ADDED TO STOP IT KEEPING ITS OWN COPY OF GIMBAL ANGLES, KEEPING IT FROM FIGHTING THE SEARCH
        self.hold = False   # True while the reacquisition search owns the gimbal
        self.create_subscription(String, "/reacq/gimbal_sync", self.on_sync, 10)

        self.timer = self.create_timer(1.0 / a.rate, self.tick)
        self.pending = None       # newest detection frame, consumed by tick()

        self.get_logger().info(f"Lock mode:  {a.lock}")
        self.get_logger().info(f"KP={a.kp}  deadzone={a.deadzone}  max_rate={a.max_rate} deg/s")
        self.get_logger().info(f"HFOV={math.degrees(a.hfov):.1f} deg")
        self.get_logger().info(f"Loss after {a.loss_timeout}s without the target")
        self.get_logger().info("Waiting for detections...")

    # ------------------------------------------------------------------
    def on_detections(self, msg: String):
        try:
            self.pending = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    ## ADDED IN TO STOP INTERFERENCE WITH REACQUISITION
    def on_sync(self, msg: String):
        """Handover from the reacquisition search: it owns the gimbal while hold is True."""
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
            
        self.hold = bool(d.get("hold", False))
        self.pitch = math.radians(d.get("pitch_deg", math.degrees(self.pitch)))  # resume from the search's angles,
        self.yaw = math.radians(d.get("yaw_deg", math.degrees(self.yaw)))        # not from where the target was lost
        self.locked_id = None                                                    # treat the next sighting as new

    def pick_target(self, dets):
        """Choose which sphere to follow, and stay with it."""
        if not dets:
            return None

        # Already locked: find that ID again.
        if self.locked_id is not None:
            for d in dets:
                if d.get("id") == self.locked_id:
                    return d
            # Track ID changed but a sphere is still visible. Slewing the
            # gimbal makes the target jump between frames, which breaks
            # ByteTrack's association and mints a new ID. With one target
            # in the scene, re-lock rather than calling it a loss.
            self.locked_id = None

        # Not locked yet - choose one and remember its ID.
        if self.a.lock == "centre":
            best = min(dets, key=lambda d: (d["centre_norm"][0] - 0.5) ** 2
                                          + (d["centre_norm"][1] - 0.5) ** 2)
        elif self.a.lock == "largest":
            best = max(dets, key=lambda d: (d["bbox_xyxy"][2] - d["bbox_xyxy"][0])
                                          * (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]))
        else:  # confidence
            best = max(dets, key=lambda d: d["conf"])

        if best.get("id") is not None:
            self.locked_id = best["id"]
            self.get_logger().info(f"Locked onto track {self.locked_id}")
        return best

    # ------------------------------------------------------------------
    def tick(self):
      ## ADDED
        if self.hold:
            return
            
        now = time.monotonic()
        frame = self.pending
        target = None

        if frame is not None:
            target = self.pick_target(frame.get("detections", []))

        if target is not None:
            self.last_seen = now
            self.declared_lost = False
            self.track(target, frame)
        else:
            self.coast(now)

    def track(self, target, frame):
        cx, cy = target["centre_norm"]

        # Vertical FOV from horizontal FOV and the frame's aspect ratio.
        vfov = 2 * math.atan(math.tan(self.a.hfov / 2) / self.a.aspect)

        ex = cx - 0.5          # + means target is right of centre
        ey = cy - 0.5          # + means target is BELOW centre (image y grows down)

        err = math.hypot(ex, ey)
        if err < self.a.deadzone:
            self.publish_status(target, ex, ey, "centred")
            return

        yaw_err = ex * self.a.hfov
        pitch_err = -ey * vfov          # target below centre -> tilt camera down

        if self.a.invert_yaw:
            yaw_err = -yaw_err
        if self.a.invert_pitch:
            pitch_err = -pitch_err

        max_step = math.radians(self.a.max_rate) / self.a.rate
        self.yaw += clamp(self.a.kp * yaw_err, -max_step, max_step)
        self.pitch += clamp(self.a.kp * pitch_err, -max_step, max_step)
        self.pitch = clamp(self.pitch, PITCH_MIN, PITCH_MAX)
        self.yaw = wrap_pi(self.yaw)

        self.pub_pitch.publish(Float64(data=self.pitch))
        self.pub_yaw.publish(Float64(data=self.yaw))

        # Keep the full state, so the moment of loss is fully described.
        self.last_state = {
            "t": time.time(),
            "track_id": self.locked_id,
            "centre_norm": [cx, cy],
            "bbox_xyxy": target["bbox_xyxy"],
            "conf": target["conf"],
            "gimbal_pitch": self.pitch,
            "gimbal_yaw": self.yaw,
            "pixel_err": [ex, ey],
        }

        self.frames += 1
        if self.frames % (self.a.rate * 2) == 0:
            self.get_logger().info(
                f"track {self.locked_id}  err={err:.3f}  "
                f"pitch={math.degrees(self.pitch):6.1f}  yaw={math.degrees(self.yaw):6.1f}")

        self.publish_status(target, ex, ey, "tracking")

    def coast(self, now):
        """Target not visible this frame. Hold position; declare loss on timeout."""
        if self.last_seen is None or self.declared_lost:
            return

        gap = now - self.last_seen
        if gap < self.a.loss_timeout:
            # A missed frame is not a loss. Hold the last command and wait.
            self.publish_status(None, 0.0, 0.0, f"no detection ({gap:.1f}s)")
            return

        self.declared_lost = True
        self.get_logger().warning(f"TARGET LOST after {gap:.1f}s without detection")

        if self.last_state:
            record = dict(self.last_state)
            record["lost_at"] = time.time()
            record["gap_s"] = round(gap, 2)
            path = Path(self.a.loss_log)
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.get_logger().warning(f"Last-known state written to {path}")
            self.get_logger().warning(
                f"  last seen at frame position {record['centre_norm']}, "
                f"gimbal pitch={math.degrees(record['gimbal_pitch']):.1f} "
                f"yaw={math.degrees(record['gimbal_yaw']):.1f} deg")

        self.publish_status(None, 0.0, 0.0, "LOST")

        if self.a.relock:
            self.get_logger().info("Releasing lock - will acquire the next target seen")
            self.locked_id = None
            self.declared_lost = False
            self.last_seen = None

    def publish_status(self, target, ex, ey, state):
        self.pub_status.publish(String(data=json.dumps({
            "state": state,
            "track_id": self.locked_id,
            "err_x": round(ex, 4),
            "err_y": round(ey, 4),
            "pitch_deg": round(math.degrees(self.pitch), 2),
            "yaw_deg": round(math.degrees(self.yaw), 2),
            "conf": target["conf"] if target else None,
        })))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kp", type=float, default=0.4,
                   help="Proportional gain. Raise for faster response, lower if it oscillates.")
    p.add_argument("--deadzone", type=float, default=0.02,
                   help="Ignore errors smaller than this fraction of the frame.")
    p.add_argument("--max-rate", type=float, default=60.0,
                   help="Maximum gimbal slew, degrees per second.")
    p.add_argument("--rate", type=int, default=20, help="Control loop Hz.")
    p.add_argument("--hfov", type=float, default=1.047,
                   help="Camera horizontal FOV in radians. Check <horizontal_fov> "
                        "in the gimbal model.sdf and set this to match.")
    p.add_argument("--aspect", type=float, default=16 / 9, help="Frame width/height.")
    p.add_argument("--lock", choices=["centre", "largest", "conf"], default="centre",
                   help="Which sphere to lock onto when none is locked yet.")
    p.add_argument("--loss-timeout", type=float, default=2.0,
                   help="Seconds without the target before declaring it lost.")
    p.add_argument("--relock", action="store_true",
                   help="After a loss, lock onto the next target seen instead of stopping.")
    p.add_argument("--loss-log", default=str(Path.home() / "loss_events.jsonl"))
    p.add_argument("--invert-yaw", action="store_true")
    p.add_argument("--invert-pitch", action="store_true")
    a = p.parse_args()

    rclpy.init()
    node = GimbalTracker(a)
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
