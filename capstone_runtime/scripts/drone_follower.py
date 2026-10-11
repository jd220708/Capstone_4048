#!/usr/bin/env python3
"""
Fly the drone autonomously behind the tracked sphere.

Replaces keyboard_offboard.py - do not run both, they would fight over the
same setpoint stream. Manual control works the same way; press F to hand over
to the follower, F again to take it back.

Run (sim, agent, bridge, detector, gimbal tracker and estimator all up):
    rosenv
    python3 drone_follower.py

    W/S  forward/back      SHIFT  up        F  toggle FOLLOW mode
    A/D  left/right        CTRL   down      X  stop, hold position
    Q/E  yaw               P      position  L  land


HOW FOLLOWING WORKS
-------------------
The estimator publishes the sphere's world position and velocity. From that:

  1. The target's heading comes from its velocity.
  2. The station to hold is STANDOFF metres behind the target along that
     heading, at FOLLOW_ALT altitude.
  3. Commanded velocity = (gain x position error) + the target's own velocity.

That second term matters. Holding station behind something moving at 5 m/s
means flying at 5 m/s - a purely proportional controller would need a
permanent position error to generate that speed, so it would always trail.
Feeding the target's velocity forward removes that lag.

Altitude and yaw are handled separately: hold FOLLOW_ALT, and point the nose
at the target so the gimbal has less work to do.


FRAMES
------
The estimator and Gazebo work in ENU (x east, y north, z up). PX4 setpoints
are NED (x north, y east, z down). Conversion happens in one place, in
send_velocity(), so there's a single spot to check if the drone flies the
wrong way.


SAFETY
------
MAX_SPEED caps commanded velocity. MIN_ALT stops it descending into the city.
If the estimate goes stale for LOSS_TIMEOUT seconds the follower disengages
and holds position rather than flying off on a guess.
"""

import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

try:
    from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                              VehicleCommand)
except ImportError:
    sys.exit("px4_msgs not found. Did you run rosenv first?")

try:
    from pynput import keyboard
except ImportError:
    sys.exit("pynput not found. pip install pynput --break-system-packages")

# Gazebo pose feed, for the drone's own position and heading
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_v_pb2 import Pose_V
except ImportError:
    try:
        from gz.transport14 import Node as GzNode
        from gz.msgs11.pose_v_pb2 import Pose_V
    except ImportError:
        sys.exit("gz transport Python bindings not found - check the imports "
                 "in keyboard_offboard.py and match them here.")

DRONE_NAME = "x500_gimbal_0"


def quat_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class DroneFollower(Node):
    def __init__(self, a):
        super().__init__("drone_follower")
        self.a = a

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub_mode = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_sp = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos)

        self.create_subscription(String, "/target_estimator/target",
                                 self.on_target, 10)

        # drone state from Gazebo (ENU)
        self.pos = None
        self.yaw = 0.0
        self.gz = GzNode()
        topic = f"/world/{a.world}/pose/info"
        if not self.gz.subscribe(Pose_V, topic, self.on_poses):
            self.get_logger().error(f"Could not subscribe to {topic}")

        # target state (ENU)
        self.tgt = None
        self.tgt_t = 0.0

        self.cmd_prev = [0.0, 0.0, 0.0]   # last commanded velocity, for slew limiting
        self.station = None               # smoothed station point
        self.tv_filt = None               # filtered target velocity
        self.manual = [0.0, 0.0, 0.0]     # east, north, up
        self.manual_yaw = 0.0
        self.follow = False
        self.keys = set()
        self.armed_at = None
        self.ticks = 0

        keyboard.Listener(on_press=self.on_press, on_release=self.on_release,
                          daemon=True).start()
        self.create_timer(1.0 / a.rate, self.tick)

        print(__doc__.split("HOW FOLLOWING")[0])
        self.get_logger().info(
            f"standoff {a.standoff} m, altitude {a.alt} m, max speed {a.max_speed} m/s")
        self.get_logger().info("Fly to where the sphere is, then press F.")

    # ---------------------------------------------------------------- input
    def on_press(self, key):
        try:
            c = key.char.lower()
        except AttributeError:
            c = {keyboard.Key.shift: "shift", keyboard.Key.ctrl: "ctrl"}.get(key)
        if c is None:
            return
        self.keys.add(c)
        if c == "f":
            self.toggle_follow()
        elif c == "x":
            self.keys.clear()
            self.follow = False
            self.get_logger().info("STOP - holding position")
        elif c == "p" and self.pos:
            self.get_logger().info(
                f"drone ({self.pos[0]:.1f}, {self.pos[1]:.1f}, {self.pos[2]:.1f})  "
                f"yaw {math.degrees(self.yaw):.0f}")
        elif c == "l":
            self.send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info("Landing")

    def on_release(self, key):
        try:
            c = key.char.lower()
        except AttributeError:
            c = {keyboard.Key.shift: "shift", keyboard.Key.ctrl: "ctrl"}.get(key)
        self.keys.discard(c)

    def toggle_follow(self):
        if self.follow:
            self.follow = False
            self.station = None
            self.tv_filt = None
            self.get_logger().info("FOLLOW OFF - manual control")
            return
        if self.tgt is None or time.monotonic() - self.tgt_t > self.a.loss_timeout:
            self.get_logger().warning(
                "No target estimate - fly until the gimbal is tracking a sphere first")
            return
        self.follow = True
        self.get_logger().info("FOLLOW ON")

    # --------------------------------------------------------------- inputs
    def on_target(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.tgt = (d["x"], d["y"], d.get("vx", 0.0), d.get("vy", 0.0))
        self.tgt_t = time.monotonic()

    def on_poses(self, msg):
        for p in msg.pose:
            if p.name == DRONE_NAME:
                q = p.orientation
                self.pos = (p.position.x, p.position.y, p.position.z)
                self.yaw = quat_yaw(q.x, q.y, q.z, q.w)
                return

    # -------------------------------------------------------------- control
    def follow_velocity(self):
        """Velocity (ENU) that holds station behind the target."""
        tx, ty, tvx, tvy = self.tgt
        dx, dy, dz = self.pos

        # Low-pass the target velocity before using it.
        #
        # Manual flight is smooth because its command is constant: hold W and
        # the setpoint doesn't change. In follow mode the setpoint is rebuilt
        # every tick from the estimated velocity, which still wobbles by about
        # 0.9 m/s. PX4 tilts to chase each wobble, and that tilt is the roll.
        # The slew limit only caps how fast the command changes; it lets every
        # wiggle through if it's slow enough. This removes the wiggles.
        if self.tv_filt is None:
            self.tv_filt = (tvx, tvy)
        else:
            c = self.a.cmd_smooth
            self.tv_filt = (c * tvx + (1 - c) * self.tv_filt[0],
                            c * tvy + (1 - c) * self.tv_filt[1])
        tvx, tvy = self.tv_filt

        speed = math.hypot(tvx, tvy)
        if speed > 0.5:
            hx, hy = tvx / speed, tvy / speed      # target heading
        else:
            # Target barely moving: sit where we are relative to it, using
            # our own bearing so the drone doesn't lurch to an arbitrary side.
            b = math.atan2(dy - ty, dx - tx)
            hx, hy = -math.cos(b), -math.sin(b)

        # Station: STANDOFF metres behind the target along its heading.
        sx = tx - hx * self.a.standoff
        sy = ty - hy * self.a.standoff

        # Smooth the station before chasing it. The position estimate is only
        # good to a few metres, so the raw station point jitters frame to
        # frame and the drone faithfully chases that jitter.
        if self.station is None:
            self.station = (sx, sy)
        else:
            b = self.a.station_smooth
            self.station = (b * sx + (1 - b) * self.station[0],
                            b * sy + (1 - b) * self.station[1])
        sx, sy = self.station

        ex, ey = sx - dx, sy - dy
        ez = self.a.alt - dz

        # Don't correct errors smaller than the estimate is accurate to -
        # that's chasing measurement noise, not real displacement. Velocity
        # feedforward still runs, so the drone keeps pace with the target.
        if math.hypot(ex, ey) < self.a.deadzone:
            ex = ey = 0.0

        # Proportional correction plus the target's own velocity. Without the
        # second term the drone would permanently trail a moving target.
        vx = clamp(self.a.kp * ex + tvx, -self.a.max_speed, self.a.max_speed)
        vy = clamp(self.a.kp * ey + tvy, -self.a.max_speed, self.a.max_speed)
        vz = clamp(self.a.kp_alt * ez, -self.a.max_climb, self.a.max_climb)

        if dz < self.a.min_alt and vz < 0:
            vz = 0.0                                # never descend below floor

        # Point the nose at the target so the gimbal has less to do.
        yaw_err = wrap_pi(math.atan2(ty - dy, tx - dx) - self.yaw)
        yawrate = clamp(self.a.kp_yaw * yaw_err,
                        -self.a.max_yawrate, self.a.max_yawrate)

        return vx, vy, vz, yawrate, math.hypot(ex, ey)

    def manual_velocity(self):
        """Body-relative WASD converted into ENU."""
        fwd = (1.0 if "w" in self.keys else 0.0) - (1.0 if "s" in self.keys else 0.0)
        rgt = (1.0 if "d" in self.keys else 0.0) - (1.0 if "a" in self.keys else 0.0)
        up = (1.0 if "shift" in self.keys else 0.0) - (1.0 if "ctrl" in self.keys else 0.0)
        yawrate = (1.0 if "e" in self.keys else 0.0) - (1.0 if "q" in self.keys else 0.0)

        s = self.a.manual_speed
        vx = (fwd * math.cos(self.yaw) - rgt * math.sin(self.yaw)) * s
        vy = (fwd * math.sin(self.yaw) + rgt * math.cos(self.yaw)) * s
        return vx, vy, up * s, yawrate * self.a.max_yawrate, 0.0

    def tick(self):
        self.ticks += 1
        self.pub_mode.publish(OffboardControlMode(
            timestamp=self.now(), position=False, velocity=True,
            acceleration=False, attitude=False, body_rate=False))

        # Arm and switch to offboard once a setpoint stream exists.
        if self.armed_at is None and self.ticks > self.a.rate:
            self.send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.armed_at = time.monotonic()
            self.get_logger().info("Armed, offboard requested - hold SHIFT to climb")

        if self.pos is None:
            return

        stale = self.tgt is None or time.monotonic() - self.tgt_t > self.a.loss_timeout
        if self.follow and stale:
            self.follow = False
            self.get_logger().warning(
                f"Target estimate stale for {self.a.loss_timeout}s - FOLLOW OFF, holding")

        if self.follow:
            vx, vy, vz, yawrate, err = self.follow_velocity()
            if self.ticks % (self.a.rate * 2) == 0:
                self.get_logger().info(
                    f"following: station error {err:6.1f} m   "
                    f"cmd ({vx:5.1f},{vy:5.1f},{vz:5.1f}) m/s")
        else:
            vx, vy, vz, yawrate, _ = self.manual_velocity()

        # Slew-limit the command. A multirotar banks in proportion to its
        # commanded acceleration, and bank angle swings the camera - which is
        # what corrupted the position estimate. Limiting acceleration keeps
        # the aircraft closer to level.
        step = self.a.max_accel / self.a.rate
        vx = clamp(vx, self.cmd_prev[0] - step, self.cmd_prev[0] + step)
        vy = clamp(vy, self.cmd_prev[1] - step, self.cmd_prev[1] + step)
        vz = clamp(vz, self.cmd_prev[2] - step, self.cmd_prev[2] + step)
        self.cmd_prev = [vx, vy, vz]

        self.send_velocity(vx, vy, vz, yawrate)

    def send_velocity(self, vx_e, vy_n, vz_u, yawrate):
        """ENU in, NED out. The only place the frames are converted."""
        sp = TrajectorySetpoint()
        sp.timestamp = self.now()
        sp.position = [float("nan")] * 3
        sp.velocity = [float(vy_n), float(vx_e), float(-vz_u)]   # N, E, D
        sp.acceleration = [float("nan")] * 3
        sp.yaw = float("nan")
        sp.yawspeed = float(-yawrate)        # NED yaw is clockwise-positive
        self.pub_sp.publish(sp)

    def send_cmd(self, command, p1=0.0, p2=0.0):
        self.pub_cmd.publish(VehicleCommand(
            timestamp=self.now(), param1=p1, param2=p2, command=command,
            target_system=1, target_component=1, source_system=1,
            source_component=1, from_external=True))

    def now(self):
        return int(self.get_clock().now().nanoseconds / 1000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="city_test")
    p.add_argument("--standoff", type=float, default=40.0,
                   help="Metres to hold behind the target.")
    p.add_argument("--alt", type=float, default=75.0,
                   help="Altitude to hold while following. Higher gives better "
                        "position estimates - see the estimator notes.")
    p.add_argument("--min-alt", type=float, default=40.0)
    p.add_argument("--kp", type=float, default=0.18,
                   help="Horizontal position gain. Low on purpose - aggressive "
                        "correction tilts the aircraft and degrades the estimate.")
    p.add_argument("--kp-alt", type=float, default=0.5)
    p.add_argument("--kp-yaw", type=float, default=1.0)
    p.add_argument("--max-speed", type=float, default=7.0,
                   help="Target moves at ~5 m/s. Higher than ~8 makes the drone "
                        "bank hard enough to disturb the camera geometry.")
    p.add_argument("--max-climb", type=float, default=3.0)
    p.add_argument("--max-accel", type=float, default=2.0,
                   help="m/s^2. Caps how fast the command may change, and so "
                        "how far the aircraft banks.")
    p.add_argument("--max-yawrate", type=float, default=0.8)
    p.add_argument("--manual-speed", type=float, default=8.0)
    p.add_argument("--deadzone", type=float, default=5.0,
                   help="Metres. Ignore station errors below this - roughly the "
                        "accuracy of the position estimate. Raise to calm jitter, "
                        "lower to hold station more tightly.")
    p.add_argument("--station-smooth", type=float, default=0.15,
                   help="Smoothing on the station point, 0-1. Lower is calmer.")
    p.add_argument("--cmd-smooth", type=float, default=0.1,
                   help="Low-pass on the target velocity, 0-1. Lower is smoother "
                        "but turns later. 0.1 is ~1 s at 10 Hz.")
    p.add_argument("--loss-timeout", type=float, default=3.0)
    p.add_argument("--rate", type=int, default=20)
    a = p.parse_args()

    rclpy.init()
    node = DroneFollower(a)
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
