#!/usr/bin/env python3

import argparse
import json
import math
import threading
import time
import os

from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ============================================================
# GAZEBO PYTHON TRANSPORT
# ============================================================

try:
    from gz.transport13 import Node as GazeboNode
    from gz.msgs10.pose_v_pb2 import Pose_V

except ImportError:

    try:
        from gz.transport14 import Node as GazeboNode
        from gz.msgs11.pose_v_pb2 import Pose_V

    except ImportError:

        raise SystemExit(
            "Gazebo Python transport bindings not found."
        )


# ============================================================
# TRACKED ENTITY
# ============================================================

@dataclass
class TrackedEntity:

    label: str
    marker: str
    color: str
    size: float

    x: float = None
    y: float = None
    z: float = None

    yaw: float = None

    vx: float = None
    vy: float = None

    updated_at: float = 0.0

    trail: deque = field(
        default_factory=lambda: deque(maxlen=600)
    )


# ============================================================
# TOP DOWN MAP
# ============================================================

class TopDownMap(Node):

    def __init__(self, args):

        super().__init__(
            "top_down_map"
        )

        self.args = args

        self.lock = threading.Lock()


        # ====================================================
        # ENTITY REGISTRY
        #
        # Add future map objects here.
        # ====================================================

        self.entities = {

            "drone": TrackedEntity(
                label="Drone",
                marker="^",
                color="tab:blue",
                size=100,
            ),

            "target_true": TrackedEntity(
                label="Target - True",
                marker="o",
                color="deeppink",
                size=90,
            ),

            "target_detected": TrackedEntity(
                label="Target - Detected",
                marker="x",
                color="tab:orange",
                size=100,
            ),
        }


        # ====================================================
        # VARIABLE REGISTRY
        #
        # Future values can be added here without changing
        # the basic map system.
        # ====================================================

        self.variables = {

            "Target Detected": "WAITING",

            "Estimation error": None,

            "Follow mode": "UNKNOWN",
        }


        # ====================================================
        # GAZEBO
        # ====================================================

        self.gz_node = GazeboNode()

        self.gz_topic = (
            f"/world/"
            f"{args.world}"
            f"/pose/info"
        )


        subscribed = self.gz_node.subscribe(

            Pose_V,

            self.gz_topic,

            self.gazebo_callback,
        )


        if subscribed:

            self.get_logger().info(

                f"Gazebo map tracking active: "
                f"{self.gz_topic}"
            )

        else:

            self.get_logger().error(

                "Could not subscribe to Gazebo poses"
            )


        # ====================================================
        # TARGET ESTIMATOR
        # ====================================================

        self.create_subscription(

            String,

            "/target_estimator/target",

            self.target_callback,

            10,
        )


        # ====================================================
        # MATPLOTLIB WINDOW
        # ====================================================

        self.fig = plt.figure(
            figsize=(12, 8)
        )


        layout = self.fig.add_gridspec(

            1,

            2,

            width_ratios=[
                4,
                1.3,
            ],
        )


        self.ax = self.fig.add_subplot(
            layout[0, 0]
        )

        # ====================================================
        # FIXED MAP BACKGROUND
        # ====================================================

        if os.path.exists(self.args.map_image):

            background = plt.imread(
                self.args.map_image
            )

            self.ax.imshow(
                background,
                extent=[
                    self.args.xmin,
                    self.args.xmax,
                    self.args.ymin,
                    self.args.ymax,
                ],
                origin="upper",
                aspect="equal",
                zorder=0,
            )

        else:

            self.get_logger().warning(
                f"Map image not found: {self.args.map_image}"
            )


        # Fixed world-coordinate limits

        self.ax.set_xlim(
            self.args.xmin,
            self.args.xmax,
        )

        self.ax.set_ylim(
            self.args.ymin,
            self.args.ymax,
        )

        self.ax.set_autoscale_on(False)

        self.info_ax = (
            self.fig.add_subplot(
                layout[0, 1]
            )
        )


        self.ax.set_title(

            "UAV Project - Top Down Tracking Map"
        )


        self.ax.set_xlabel(

            "World X (m)"
        )


        self.ax.set_ylabel(

            "World Y (m)"
        )


        self.ax.set_aspect(

            "equal",

            adjustable="box",
        )


        self.ax.grid(

            True,

            alpha=0.3,
        )


        self.info_ax.axis(
            "off"
        )


        # ====================================================
        # MAP OBJECTS
        # ====================================================

        self.points = {}

        self.trails = {}


        for name, entity in self.entities.items():

            self.points[name] = (
                self.ax.scatter(

                    [],

                    [],

                    s=entity.size,

                    marker=entity.marker,

                    c=entity.color,

                    label=entity.label,

                    zorder=5,
                )
            )


            line, = self.ax.plot(

                [],

                [],

                color=entity.color,

                linewidth=1.2,

                alpha=0.5,
            )


            self.trails[name] = line


        # Drone heading

        self.heading_line, = (
            self.ax.plot(

                [],

                [],

                color="tab:blue",

                linewidth=2,
            )
        )


        # Target velocity vector

        self.velocity_line, = (
            self.ax.plot(

                [],

                [],

                color="tab:orange",

                linewidth=2,
            )
        )


        # Estimation-error line

        self.error_line, = (
            self.ax.plot(

                [],

                [],

                linestyle="--",

                color="gray",

                linewidth=1.2,
            )
        )


        self.ax.legend(

            loc="upper right"
        )


        self.info_text = (
            self.info_ax.text(

                0.02,

                0.98,

                "Waiting for data...",

                va="top",

                family="monospace",

                fontsize=10,
            )
        )

        # ====================================================
        # GUI UPDATE
        # ====================================================

        self.animation = FuncAnimation(

            self.fig,

            self.update_map,

            interval=100,

            cache_frame_data=False,
        )


    # ========================================================
    # QUATERNION -> YAW
    # ========================================================

    def quaternion_yaw(self, q):

        return math.atan2(

            2.0
            *
            (
                q.w * q.z
                +
                q.x * q.y
            ),

            1.0
            -
            2.0
            *
            (
                q.y * q.y
                +
                q.z * q.z
            ),
        )


    # ========================================================
    # UPDATE ENTITY
    # ========================================================

    def update_entity(

        self,

        name,

        x,

        y,

        z=None,

        yaw=None,

        vx=None,

        vy=None,
    ):

        entity = (
            self.entities[name]
        )


        entity.x = float(x)

        entity.y = float(y)


        if z is not None:

            entity.z = float(z)


        if yaw is not None:

            entity.yaw = float(yaw)


        if vx is not None:

            entity.vx = float(vx)


        if vy is not None:

            entity.vy = float(vy)


        entity.updated_at = (
            time.monotonic()
        )


        entity.trail.append(

            (
                entity.x,

                entity.y,
            )
        )


    # ========================================================
    # GAZEBO CALLBACK
    #
    # True drone position
    # True pink target position
    # ========================================================

    def gazebo_callback(self, msg):

        with self.lock:

            for pose in msg.pose:

                name = (
                    pose.name
                    .split("::")[-1]
                )


                # ----------------------------------------
                # DRONE
                # ----------------------------------------

                if name == "x500_gimbal_0":

                    self.update_entity(

                        "drone",

                        pose.position.x,

                        pose.position.y,

                        pose.position.z,

                        yaw=self.quaternion_yaw(
                            pose.orientation
                        ),
                    )


                # ----------------------------------------
                # PINK TARGET
                #
                # Sphere plugin generates names:
                #
                # sphere_pink_0
                # sphere_pink_1
                # ...
                # ----------------------------------------

                elif "pink" in name.lower():

                    self.update_entity(
                        "target_true",
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                    )


    # ========================================================
    # DETECTED TARGET CALLBACK
    # ========================================================

    def target_callback(self, msg):

        try:

            data = json.loads(
                msg.data
            )


            x = data["x"]

            y = data["y"]


        except Exception:

            return


        with self.lock:

            self.update_entity(

                "target_detected",

                x,

                y,

                data.get("z"),

                vx=data.get("vx"),

                vy=data.get("vy"),
            )


            self.variables[
                "Target Detected"
            ] = "LIVE"


    # ========================================================
    # DRAW MAP
    # ========================================================

    def update_map(self, frame):

        rclpy.spin_once(
            self,
            timeout_sec=0.0
        )

        with self.lock:

            # --------------------------------------------
            # DRAW ENTITIES
            # --------------------------------------------

            for name, entity in (
                self.entities.items()
            ):


                if (
                    entity.x is None
                    or
                    entity.y is None
                ):

                    continue


                self.points[
                    name
                ].set_offsets(

                    [[
                        entity.x,

                        entity.y,
                    ]]
                )


                if entity.trail:

                    xs = [

                        p[0]

                        for p in
                        entity.trail
                    ]


                    ys = [

                        p[1]

                        for p in
                        entity.trail
                    ]


                    self.trails[
                        name
                    ].set_data(

                        xs,

                        ys,
                    )


            # ============================================
            # DRONE HEADING
            # ============================================

            drone = (
                self.entities["drone"]
            )


            if (
                drone.x is not None

                and

                drone.yaw is not None
            ):

                length = 20.0


                x2 = (

                    drone.x

                    +

                    length
                    *
                    math.cos(
                        drone.yaw
                    )
                )


                y2 = (

                    drone.y

                    +

                    length
                    *
                    math.sin(
                        drone.yaw
                    )
                )


                self.heading_line.set_data(

                    [
                        drone.x,

                        x2,
                    ],

                    [
                        drone.y,

                        y2,
                    ],
                )


            # ============================================
            # TARGET VELOCITY VECTOR
            # ============================================

            target_detected = (
                self.entities[
                    "target_detected"
                ]
            )


            if (

                target_detected.x
                is not None

                and

                target_detected.vx
                is not None

                and

                target_detected.vy
                is not None
            ):

                seconds = 3.0


                self.velocity_line.set_data(

                    [
                        target_detected.x,

                        target_detected.x
                        +
                        target_detected.vx
                        *
                        seconds,
                    ],

                    [
                        target_detected.y,

                        target_detected.y
                        +
                        target_detected.vy
                        *
                        seconds,
                    ],
                )


            # ============================================
            # ESTIMATION ERROR
            # ============================================

            true_target = (
                self.entities[
                    "target_true"
                ]
            )


            if (

                true_target.x
                is not None

                and

                target_detected.x
                is not None
            ):

                error = (
                    math.hypot(

                        true_target.x
                        -
                        target_detected.x,

                        true_target.y
                        -
                        target_detected.y,
                    )
                )


                self.variables[
                    "Estimation error"
                ] = error


                self.error_line.set_data(

                    [
                        true_target.x,

                        target_detected.x,
                    ],

                    [
                        true_target.y,

                        target_detected.y,
                    ],
                )

            # ============================================
            # TELEMETRY PANEL
            # ============================================

            lines = [

                "LIVE TELEMETRY",

                "==============",

                "",
            ]


            for name in [

                "drone",

                "target_true",

                "target_detected",
            ]:

                entity = (
                    self.entities[name]
                )


                lines.append(
                    entity.label
                )


                if entity.x is None:

                    lines.append(

                        "  waiting..."
                    )

                else:

                    lines.append(

                        f"  X: "
                        f"{entity.x:8.2f} m"
                    )


                    lines.append(

                        f"  Y: "
                        f"{entity.y:8.2f} m"
                    )


                    if (
                        entity.z
                        is not None
                    ):

                        lines.append(

                            f"  Z: "
                            f"{entity.z:8.2f} m"
                        )


                    if (

                        name
                        ==
                        "target_detected"

                        and

                        entity.vx
                        is not None
                    ):

                        speed = (
                            math.hypot(

                                entity.vx,

                                entity.vy,
                            )
                        )


                        lines.append(

                            f"  V: "
                            f"{speed:8.2f} m/s"
                        )


                lines.append("")


            # ============================================
            # EXTRA VARIABLES
            # ============================================

            lines.append(
                "VARIABLES"
            )

            lines.append(
                "========="
            )


            for (
                name,

                value,

            ) in self.variables.items():


                if (
                    isinstance(
                        value,
                        float,
                    )
                ):

                    value = (
                        f"{value:.2f}"
                    )


                lines.append(

                    f"{name}: "
                    f"{value}"
                )


            self.info_text.set_text(

                "\n".join(
                    lines
                )
            )


        return []


# ============================================================
# MAIN
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
        "--map-image",
        default=os.path.expanduser(
            "~/Documents/Simulation_City/capstone_runtime/map_system/city_topdown.png"
        ),
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=1556.0,
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=1573.0,
    )


    args = (
        parser.parse_args()
    )


    rclpy.init()


    node = (
        TopDownMap(
            args
        )
    )


    try:

        plt.tight_layout()

        plt.show()


    except KeyboardInterrupt:

        pass


    finally:

        if rclpy.ok():

            rclpy.shutdown()


        node.destroy_node()


if __name__ == "__main__":

    main()
