#!/usr/bin/env python3

import argparse
import json
import math
import sys
import threading
import time

import rclpy

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)

from std_msgs.msg import String, Float64


try:
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
    )

except ImportError:
    sys.exit(
        "px4_msgs not found. "
        "Source ROS 2 and ~/ros2_ws/install/setup.bash."
    )


try:
    from pynput import keyboard

except ImportError:
    sys.exit(
        "pynput not found."
    )


# ============================================================
# Gazebo transport
# ============================================================

try:

    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_v_pb2 import Pose_V

except ImportError:

    try:

        from gz.transport14 import Node as GzNode
        from gz.msgs11.pose_v_pb2 import Pose_V

    except ImportError:

        sys.exit(
            "Gazebo Python transport bindings not found."
        )


DRONE_NAME = "x500_gimbal_0"


# ============================================================
# Helpers
# ============================================================

def quat_yaw(x, y, z, w):

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def clamp(value, low, high):

    return max(
        low,
        min(high, value),
    )


def wrap_pi(angle):

    return (
        angle + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


# ============================================================
# Drone follower
# ============================================================

class DroneFollower(Node):

    def __init__(self, args):

        super().__init__(
            "drone_follower"
        )

        self.a = args


        # ====================================================
        # PX4 QoS
        # ====================================================

        qos = QoSProfile(

            reliability=
            ReliabilityPolicy.BEST_EFFORT,

            durability=
            DurabilityPolicy.TRANSIENT_LOCAL,

            history=
            HistoryPolicy.KEEP_LAST,

            depth=1,
        )


        # ====================================================
        # PX4 publishers
        # ====================================================

        self.pub_mode = self.create_publisher(

            OffboardControlMode,

            "/fmu/in/offboard_control_mode",

            qos,
        )


        self.pub_sp = self.create_publisher(

            TrajectorySetpoint,

            "/fmu/in/trajectory_setpoint",

            qos,
        )


        self.pub_cmd = self.create_publisher(

            VehicleCommand,

            "/fmu/in/vehicle_command",

            qos,
        )


        # ====================================================
        # Gimbal publishers
        # ====================================================

        self.gimbal_pitch_pub = (
            self.create_publisher(

                Float64,

                "/model/x500_gimbal_0/"
                "command/gimbal_pitch_demo",

                10,
            )
        )


        self.gimbal_yaw_pub = (
            self.create_publisher(

                Float64,

                "/model/x500_gimbal_0/"
                "command/gimbal_yaw_demo",

                10,
            )
        )


        # ====================================================
        # Camera zoom publisher
        # ====================================================

        self.zoom_pub = (
            self.create_publisher(

                Float64,

                "/model/x500_gimbal_0/"
                "camera/zoom/cmd_zoom",

                10,
            )
        )


        # ====================================================
        # Target estimator
        # ====================================================

        self.create_subscription(

            String,

            "/target_estimator/target",

            self.on_target,

            10,
        )


        self.tgt = None

        self.tgt_t = 0.0


        # ====================================================
        # Gazebo drone pose
        # ====================================================

        self.pos = None

        self.yaw = 0.0


        self.gz = GzNode()


        pose_topic = (

            f"/world/"
            f"{self.a.world}"
            f"/pose/info"
        )


        subscribed = (
            self.gz.subscribe(

                Pose_V,

                pose_topic,

                self.on_poses,
            )
        )


        if not subscribed:

            self.get_logger().error(

                f"Could not subscribe "
                f"to {pose_topic}"
            )


        # ====================================================
        # Follow controller state
        # ====================================================

        self.follow = False

        self.station = None

        self.tv_filt = None


        self.cmd_prev = [
            0.0,
            0.0,
            0.0,
        ]


        self.armed_at = None

        self.ticks = 0


        # ====================================================
        # Keyboard state
        # ====================================================

        self.keys = set()

        self.key_lock = (
            threading.Lock()
        )


        # Key latches prevent repeated
        # actions while a key is held.

        self.position_key_latched = False

        self.follow_key_latched = False

        self.land_key_latched = False

        self.zoom_in_latched = False

        self.zoom_out_latched = False


        # ====================================================
        # Manual gimbal
        # ====================================================

        self.gimbal_pitch = 0.0

        self.gimbal_yaw = 0.0


        self.gimbal_rate = (
            math.radians(90.0)
        )


        self.gimbal_pitch_min = (
            math.radians(-135.0)
        )


        self.gimbal_pitch_max = (
            math.radians(45.0)
        )


        # ====================================================
        # Zoom levels
        # ====================================================

        self.zoom_levels = [

            1.0,

            1.5,

            2.0,

            3.0,

            4.0,

            6.0,

            8.0,
        ]


        self.zoom_index = 0


        # ====================================================
        # Keyboard listener
        # ====================================================

        self.listener = (
            keyboard.Listener(

                on_press=
                self.on_press,

                on_release=
                self.on_release,
            )
        )


        self.listener.start()


        # ====================================================
        # Main loop
        # ====================================================

        self.create_timer(

            1.0 / self.a.rate,

            self.tick,
        )


        self.print_controls()


    # ========================================================
    # Controls display
    # ========================================================

    def print_controls(self):

        print()

        print(
            "==============================="
        )

        print(
            " PX4 FOLLOWER CONTROL"
        )

        print(
            "==============================="
        )

        print(
            " W / S       Forward / Backward"
        )

        print(
            " A / D       Left / Right"
        )

        print(
            " Q / E       Yaw Left / Right"
        )

        print(
            " SHIFT       Up"
        )

        print(
            " CTRL        Down"
        )

        print(
            " X           Stop / Cancel Follow"
        )

        print()

        print(
            " UP / DOWN   Gimbal Tilt"
        )

        print(
            " LEFT/RIGHT  Gimbal Pan"
        )

        print(
            " [           Zoom Out"
        )

        print(
            " ]           Zoom In"
        )

        print()

        print(
            " P           Print Gazebo XYZ"
        )

        print(
            " F           Toggle FOLLOW"
        )

        print(
            " L           Land"
        )

        print(
            "==============================="
        )

        print()


        self.get_logger().info(

            f"Manual speed: "
            f"{self.a.manual_speed:.1f} m/s | "

            f"climb: "
            f"{self.a.manual_climb:.1f} m/s | "

            f"yaw: "
            f"{math.degrees(self.a.manual_yawrate):.0f} deg/s"
        )


        self.get_logger().info(

            f"Follow: standoff "
            f"{self.a.standoff:.1f} m | "

            f"altitude "
            f"{self.a.alt:.1f} m | "

            f"max speed "
            f"{self.a.max_speed:.1f} m/s"
        )


    # ========================================================
    # Keyboard helpers
    # ========================================================

    def normalise_key(self, key):

        try:

            return key.char.lower()

        except AttributeError:

            return key


    def pressed(self, key):

        with self.key_lock:

            return (
                key in self.keys
            )


    # ========================================================
    # Keyboard press
    # ========================================================

    def on_press(self, key):

        k = self.normalise_key(
            key
        )


        # -----------------------------------------------
        # P = print Gazebo position
        # -----------------------------------------------

        if k == "p":

            if not self.position_key_latched:

                self.position_key_latched = True

                self.print_position()

            return


        # -----------------------------------------------
        # F = FOLLOW toggle
        # -----------------------------------------------

        if k == "f":

            if not self.follow_key_latched:

                self.follow_key_latched = True

                self.toggle_follow()

            return


        # -----------------------------------------------
        # L = land
        # -----------------------------------------------

        if k == "l":

            if not self.land_key_latched:

                self.land_key_latched = True

                self.follow = False


                self.send_cmd(

                    VehicleCommand.
                    VEHICLE_CMD_NAV_LAND
                )


                self.get_logger().info(
                    "Landing"
                )

            return


        # -----------------------------------------------
        # ] = zoom in
        # -----------------------------------------------

        if k == "]":

            if not self.zoom_in_latched:

                self.zoom_in_latched = True


                if (
                    self.zoom_index
                    <
                    len(self.zoom_levels) - 1
                ):

                    self.zoom_index += 1


                self.publish_zoom()

            return


        # -----------------------------------------------
        # [ = zoom out
        # -----------------------------------------------

        if k == "[":

            if not self.zoom_out_latched:

                self.zoom_out_latched = True


                if self.zoom_index > 0:

                    self.zoom_index -= 1


                self.publish_zoom()

            return


        # -----------------------------------------------
        # X = immediate stop
        # -----------------------------------------------

        if (
            k == "x"
            and self.follow
        ):

            self.follow = False

            self.station = None

            self.tv_filt = None


            self.cmd_prev = [
                0.0,
                0.0,
                0.0,
            ]


            self.get_logger().info(

                "FOLLOW OFF - STOP"
            )


        # Normal held key

        with self.key_lock:

            self.keys.add(k)


    # ========================================================
    # Keyboard release
    # ========================================================

    def on_release(self, key):

        k = self.normalise_key(
            key
        )


        if k == "p":

            self.position_key_latched = False

            return


        if k == "f":

            self.follow_key_latched = False

            return


        if k == "l":

            self.land_key_latched = False

            return


        if k == "]":

            self.zoom_in_latched = False

            return


        if k == "[":

            self.zoom_out_latched = False

            return


        with self.key_lock:

            self.keys.discard(k)


    # ========================================================
    # Position display
    # ========================================================

    def print_position(self):

        if self.pos is None:

            print()

            print(
                "Drone position not available yet."
            )

            print()

            return


        x, y, z = self.pos


        print()

        print(
            "=============================="
        )

        print(
            " DRONE POSITION"
        )

        print(
            "=============================="
        )

        print(
            f"X = {x:.3f}"
        )

        print(
            f"Y = {y:.3f}"
        )

        print(
            f"Z = {z:.3f}"
        )

        print(
            "=============================="
        )

        print()


    # ========================================================
    # Camera zoom
    # ========================================================

    def publish_zoom(self):

        zoom = self.zoom_levels[
            self.zoom_index
        ]


        msg = Float64()

        msg.data = float(
            zoom
        )


        self.zoom_pub.publish(
            msg
        )


        self.get_logger().info(

            f"Camera zoom: "
            f"{zoom:.1f}x"
        )


    # ========================================================
    # Target estimator callback
    # ========================================================

    def on_target(self, msg):

        try:

            data = json.loads(
                msg.data
            )

        except json.JSONDecodeError:

            return


        self.tgt = (

            data["x"],

            data["y"],

            data.get(
                "vx",
                0.0,
            ),

            data.get(
                "vy",
                0.0,
            ),
        )


        self.tgt_t = (
            time.monotonic()
        )


    # ========================================================
    # Gazebo pose callback
    # ========================================================

    def on_poses(self, msg):

        for pose in msg.pose:

            if (

                pose.name
                ==
                DRONE_NAME

                or

                pose.name.endswith(

                    f"::{DRONE_NAME}"
                )
            ):

                q = (
                    pose.orientation
                )


                self.pos = (

                    float(
                        pose.position.x
                    ),

                    float(
                        pose.position.y
                    ),

                    float(
                        pose.position.z
                    ),
                )


                self.yaw = quat_yaw(

                    q.x,

                    q.y,

                    q.z,

                    q.w,
                )


                return


    # ========================================================
    # FOLLOW toggle
    # ========================================================

    def toggle_follow(self):

        # Turn FOLLOW off

        if self.follow:

            self.follow = False

            self.station = None

            self.tv_filt = None


            self.get_logger().info(

                "FOLLOW OFF - manual control"
            )


            return


        # Don't start without
        # a current target estimate.

        if (

            self.tgt is None

            or

            time.monotonic()
            -
            self.tgt_t
            >
            self.a.loss_timeout
        ):

            self.get_logger().warning(

                "No target estimate - "
                "wait until the target "
                "estimator is tracking."
            )


            return


        self.follow = True


        self.get_logger().info(
            "FOLLOW ON"
        )


    # ========================================================
    # Autonomous follower
    # ========================================================

    def follow_velocity(self):

        (
            tx,
            ty,
            tvx,
            tvy,
        ) = self.tgt


        (
            dx,
            dy,
            dz,
        ) = self.pos


        # -----------------------------------------------
        # Smooth target velocity
        # -----------------------------------------------

        if self.tv_filt is None:

            self.tv_filt = (
                tvx,
                tvy,
            )

        else:

            c = (
                self.a.cmd_smooth
            )


            self.tv_filt = (

                c * tvx

                +

                (1.0 - c)
                *
                self.tv_filt[0],


                c * tvy

                +

                (1.0 - c)
                *
                self.tv_filt[1],
            )


        (
            tvx,
            tvy,
        ) = self.tv_filt


        target_speed = (
            math.hypot(
                tvx,
                tvy,
            )
        )


        # -----------------------------------------------
        # Target heading
        # -----------------------------------------------

        if target_speed > 0.5:

            heading_x = (
                tvx
                /
                target_speed
            )


            heading_y = (
                tvy
                /
                target_speed
            )

        else:

            bearing = (
                math.atan2(

                    dy - ty,

                    dx - tx,
                )
            )


            heading_x = (
                -math.cos(
                    bearing
                )
            )


            heading_y = (
                -math.sin(
                    bearing
                )
            )


        # -----------------------------------------------
        # Desired station point behind target
        # -----------------------------------------------

        station_x = (

            tx

            -

            heading_x
            *
            self.a.standoff
        )


        station_y = (

            ty

            -

            heading_y
            *
            self.a.standoff
        )


        # -----------------------------------------------
        # Smooth station point
        # -----------------------------------------------

        if self.station is None:

            self.station = (
                station_x,
                station_y,
            )

        else:

            b = (
                self.a.station_smooth
            )


            self.station = (

                b * station_x

                +

                (1.0 - b)
                *
                self.station[0],


                b * station_y

                +

                (1.0 - b)
                *
                self.station[1],
            )


        (
            station_x,
            station_y,
        ) = self.station


        # -----------------------------------------------
        # Position errors
        # -----------------------------------------------

        error_x = (
            station_x - dx
        )


        error_y = (
            station_y - dy
        )


        error_z = (
            self.a.alt - dz
        )


        horizontal_error = (
            math.hypot(

                error_x,

                error_y,
            )
        )


        # Ignore very small
        # position-estimate noise.

        if (
            horizontal_error
            <
            self.a.deadzone
        ):

            error_x = 0.0

            error_y = 0.0


        # -----------------------------------------------
        # Horizontal follow velocity
        # -----------------------------------------------

        vx = clamp(

            self.a.kp
            *
            error_x

            +

            tvx,

            -self.a.max_speed,

            self.a.max_speed,
        )


        vy = clamp(

            self.a.kp
            *
            error_y

            +

            tvy,

            -self.a.max_speed,

            self.a.max_speed,
        )


        # -----------------------------------------------
        # Altitude
        # -----------------------------------------------

        vz = clamp(

            self.a.kp_alt
            *
            error_z,

            -self.a.max_climb,

            self.a.max_climb,
        )


        if (

            dz
            <
            self.a.min_alt

            and

            vz < 0.0
        ):

            vz = 0.0


        # -----------------------------------------------
        # Point drone toward target
        # -----------------------------------------------

        desired_yaw = (
            math.atan2(

                ty - dy,

                tx - dx,
            )
        )


        yaw_error = wrap_pi(

            desired_yaw
            -
            self.yaw
        )


        yawrate = clamp(

            self.a.kp_yaw
            *
            yaw_error,

            -self.a.max_yawrate,

            self.a.max_yawrate,
        )


        return (

            vx,

            vy,

            vz,

            yawrate,

            horizontal_error,
        )


    # ========================================================
    # Manual flight
    # ========================================================

    def manual_velocity(self):

        forward = 0.0

        right = 0.0

        up = 0.0

        yawrate = 0.0


        # -----------------------------------------------
        # Forward / backward
        # -----------------------------------------------

        if self.pressed("w"):

            forward += (
                self.a.manual_speed
            )


        if self.pressed("s"):

            forward -= (
                self.a.manual_speed
            )


        # -----------------------------------------------
        # Left / right
        # -----------------------------------------------

        if self.pressed("d"):

            right += (
                self.a.manual_speed
            )


        if self.pressed("a"):

            right -= (
                self.a.manual_speed
            )


        # -----------------------------------------------
        # Up / down
        # -----------------------------------------------

        if (

            self.pressed(
                keyboard.Key.shift
            )

            or

            self.pressed(
                keyboard.Key.shift_l
            )

            or

            self.pressed(
                keyboard.Key.shift_r
            )
        ):

            up += (
                self.a.manual_climb
            )


        if (

            self.pressed(
                keyboard.Key.ctrl
            )

            or

            self.pressed(
                keyboard.Key.ctrl_l
            )

            or

            self.pressed(
                keyboard.Key.ctrl_r
            )
        ):

            up -= (
                self.a.manual_climb
            )


        # -----------------------------------------------
        # Yaw
        #
        # Q = left
        # E = right
        # -----------------------------------------------

        if self.pressed("q"):

            yawrate += (
                self.a.manual_yawrate
            )


        if self.pressed("e"):

            yawrate -= (
                self.a.manual_yawrate
            )


        # -----------------------------------------------
        # X = immediate stop
        # -----------------------------------------------

        if self.pressed("x"):

            return (

                0.0,

                0.0,

                0.0,

                0.0,

                0.0,
            )


        # -----------------------------------------------
        # Body-relative motion -> Gazebo ENU
        # -----------------------------------------------

        vx = (

            forward
            *
            math.cos(
                self.yaw
            )

            -

            right
            *
            math.sin(
                self.yaw
            )
        )


        vy = (

            forward
            *
            math.sin(
                self.yaw
            )

            +

            right
            *
            math.cos(
                self.yaw
            )
        )


        return (

            vx,

            vy,

            up,

            yawrate,

            0.0,
        )


    # ========================================================
    # Manual gimbal control
    # ========================================================

    def manual_gimbal_control(self):

        changed = False


        dt = (
            1.0
            /
            self.a.rate
        )


        # Up arrow

        if self.pressed(
            keyboard.Key.up
        ):

            self.gimbal_pitch += (

                self.gimbal_rate
                *
                dt
            )

            changed = True


        # Down arrow

        if self.pressed(
            keyboard.Key.down
        ):

            self.gimbal_pitch -= (

                self.gimbal_rate
                *
                dt
            )

            changed = True


        # Left arrow

        if self.pressed(
            keyboard.Key.left
        ):

            self.gimbal_yaw -= (

                self.gimbal_rate
                *
                dt
            )

            changed = True


        # Right arrow

        if self.pressed(
            keyboard.Key.right
        ):

            self.gimbal_yaw += (

                self.gimbal_rate
                *
                dt
            )

            changed = True


        # Pitch limits

        self.gimbal_pitch = max(

            self.gimbal_pitch_min,

            min(

                self.gimbal_pitch_max,

                self.gimbal_pitch,
            ),
        )


        if changed:

            pitch_msg = Float64()

            pitch_msg.data = float(
                self.gimbal_pitch
            )


            self.gimbal_pitch_pub.publish(
                pitch_msg
            )


            yaw_msg = Float64()

            yaw_msg.data = float(
                self.gimbal_yaw
            )


            self.gimbal_yaw_pub.publish(
                yaw_msg
            )


    # ========================================================
    # Main control loop
    # ========================================================

    def tick(self):

        self.ticks += 1


        # -----------------------------------------------
        # Offboard heartbeat
        # -----------------------------------------------

        mode = OffboardControlMode()


        mode.timestamp = (
            self.now()
        )


        mode.position = False

        mode.velocity = True

        mode.acceleration = False

        mode.attitude = False

        mode.body_rate = False


        self.pub_mode.publish(
            mode
        )


        # -----------------------------------------------
        # Enter Offboard + arm
        # -----------------------------------------------

        if (

            self.armed_at is None

            and

            self.ticks
            >
            self.a.rate
        ):

            self.send_cmd(

                VehicleCommand.
                VEHICLE_CMD_DO_SET_MODE,

                1.0,

                6.0,
            )


            self.send_cmd(

                VehicleCommand.
                VEHICLE_CMD_COMPONENT_ARM_DISARM,

                1.0,
            )


            self.armed_at = (
                time.monotonic()
            )


            self.get_logger().info(

                "Armed, offboard requested - "
                "hold SHIFT to climb"
            )


        if self.pos is None:

            return


        # -----------------------------------------------
        # Target loss safety
        # -----------------------------------------------

        target_stale = (

            self.tgt is None

            or

            time.monotonic()
            -
            self.tgt_t
            >
            self.a.loss_timeout
        )


        if (

            self.follow

            and

            target_stale
        ):

            self.follow = False

            self.station = None

            self.tv_filt = None


            self.get_logger().warning(

                "Target estimate stale - "
                "FOLLOW OFF"
            )


        # -----------------------------------------------
        # FOLLOW mode
        # -----------------------------------------------

        if self.follow:

            (

                vx,

                vy,

                vz,

                yawrate,

                error,

            ) = (
                self.follow_velocity()
            )


            if (

                self.ticks

                %

                (
                    self.a.rate
                    *
                    2
                )

                == 0
            ):

                self.get_logger().info(

                    f"following: "

                    f"station error "
                    f"{error:.1f} m | "

                    f"cmd "
                    f"({vx:.1f}, "
                    f"{vy:.1f}, "
                    f"{vz:.1f}) m/s"
                )


        # -----------------------------------------------
        # Manual mode
        # -----------------------------------------------

        else:

            (

                vx,

                vy,

                vz,

                yawrate,

                _,

            ) = (
                self.manual_velocity()
            )


            self.manual_gimbal_control()


        # -----------------------------------------------
        # X = hard stop
        # -----------------------------------------------

        hard_stop = (

            self.pressed("x")

            and

            not self.follow
        )


        if hard_stop:

            vx = 0.0

            vy = 0.0

            vz = 0.0

            yawrate = 0.0


            self.cmd_prev = [

                0.0,

                0.0,

                0.0,
            ]


        # -----------------------------------------------
        # Normal acceleration limiting
        # -----------------------------------------------

        else:

            step = (

                self.a.max_accel

                /

                self.a.rate
            )


            vx = clamp(

                vx,

                self.cmd_prev[0]
                -
                step,

                self.cmd_prev[0]
                +
                step,
            )


            vy = clamp(

                vy,

                self.cmd_prev[1]
                -
                step,

                self.cmd_prev[1]
                +
                step,
            )


            vz = clamp(

                vz,

                self.cmd_prev[2]
                -
                step,

                self.cmd_prev[2]
                +
                step,
            )


            self.cmd_prev = [

                vx,

                vy,

                vz,
            ]


        self.send_velocity(

            vx,

            vy,

            vz,

            yawrate,
        )


    # ========================================================
    # Velocity output
    # ========================================================

    def send_velocity(

        self,

        vx_east,

        vy_north,

        vz_up,

        yawrate_enu,
    ):

        msg = (
            TrajectorySetpoint()
        )


        msg.timestamp = (
            self.now()
        )


        nan = float("nan")


        msg.position = [

            nan,

            nan,

            nan,
        ]


        # -----------------------------------------------
        # Gazebo ENU -> PX4 NED
        # -----------------------------------------------

        msg.velocity = [

            float(
                vy_north
            ),

            float(
                vx_east
            ),

            float(
                -vz_up
            ),
        ]


        msg.acceleration = [

            nan,

            nan,

            nan,
        ]


        msg.yaw = nan


        # ENU yaw is CCW-positive.
        # PX4 NED yaw is clockwise-positive.

        msg.yawspeed = float(

            -yawrate_enu
        )


        self.pub_sp.publish(
            msg
        )


    # ========================================================
    # Vehicle command
    # ========================================================

    def send_cmd(

        self,

        command,

        param1=0.0,

        param2=0.0,
    ):

        msg = VehicleCommand()


        msg.timestamp = (
            self.now()
        )


        msg.param1 = float(
            param1
        )


        msg.param2 = float(
            param2
        )


        msg.command = (
            command
        )


        msg.target_system = 1

        msg.target_component = 1


        msg.source_system = 1

        msg.source_component = 1


        msg.from_external = True


        self.pub_cmd.publish(
            msg
        )


    # ========================================================
    # PX4 timestamp
    # ========================================================

    def now(self):

        return int(

            self.get_clock()
            .now()
            .nanoseconds

            /

            1000
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(

        "--world",

        default="city_test",
    )


    parser.add_argument(

        "--standoff",

        type=float,

        default=40.0,
    )


    parser.add_argument(

        "--alt",

        type=float,

        default=75.0,
    )


    parser.add_argument(

        "--min-alt",

        type=float,

        default=40.0,
    )


    parser.add_argument(

        "--kp",

        type=float,

        default=0.18,
    )


    parser.add_argument(

        "--kp-alt",

        type=float,

        default=0.5,
    )


    parser.add_argument(

        "--kp-yaw",

        type=float,

        default=1.0,
    )


    parser.add_argument(

        "--max-speed",

        type=float,

        default=7.0,
    )


    parser.add_argument(

        "--max-climb",

        type=float,

        default=3.0,
    )


    parser.add_argument(

        "--max-accel",

        type=float,

        default=2.0,
    )


    parser.add_argument(

        "--max-yawrate",

        type=float,

        default=0.8,
    )


    # ========================================================
    # Manual control settings
    #
    # Matches keyboard_offboard.py
    # ========================================================

    parser.add_argument(

        "--manual-speed",

        type=float,

        default=10.0,
    )


    parser.add_argument(

        "--manual-climb",

        type=float,

        default=5.0,
    )


    parser.add_argument(

        "--manual-yawrate",

        type=float,

        default=
        math.radians(90.0),
    )


    # ========================================================
    # Follow tuning
    # ========================================================

    parser.add_argument(

        "--deadzone",

        type=float,

        default=5.0,
    )


    parser.add_argument(

        "--station-smooth",

        type=float,

        default=0.15,
    )


    parser.add_argument(

        "--cmd-smooth",

        type=float,

        default=0.1,
    )


    parser.add_argument(

        "--loss-timeout",

        type=float,

        default=3.0,
    )


    parser.add_argument(

        "--rate",

        type=int,

        default=20,
    )


    args = (
        parser.parse_args()
    )


    rclpy.init()


    node = DroneFollower(
        args
    )


    try:

        rclpy.spin(
            node
        )


    except KeyboardInterrupt:

        pass


    finally:

        node.listener.stop()


        node.destroy_node()


        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()