#!/usr/bin/env python3

import argparse
import json
import math
import os
import threading
import time

from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import rclpy

from rclpy.node import Node

from std_msgs.msg import String


# ============================================================
# GAZEBO TRANSPORT
# ============================================================

try:

    from gz.transport13 import (
        Node as GazeboNode
    )

    from gz.msgs10.pose_v_pb2 import (
        Pose_V
    )

    from gz.msgs10.boolean_pb2 import (
        Boolean
    )

except ImportError:

    from gz.transport14 import (
        Node as GazeboNode
    )

    from gz.msgs11.pose_v_pb2 import (
        Pose_V
    )

    from gz.msgs11.boolean_pb2 import (
        Boolean
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
        default_factory=lambda:
        deque(
            maxlen=600
        )
    )


# ============================================================
# TOP DOWN MAP
# ============================================================

class TopDownMap(Node):

    def __init__(
        self,
        args
    ):

        super().__init__(
            "top_down_map"
        )


        self.args = args

        self.lock = (
            threading.Lock()
        )


        # ====================================================
        # ENTITIES
        # ====================================================

        self.entities = {

            "drone":

            TrackedEntity(
                label="Drone",
                marker="^",
                color="tab:blue",
                size=100,
            ),


            "target_true":

            TrackedEntity(
                label="Target - True",
                marker="o",
                color="deeppink",
                size=90,
            ),


            "target_detected":

            TrackedEntity(
                label="Target - Detected",
                marker="x",
                color="tab:orange",
                size=100,
            ),
        }


        # ====================================================
        # VARIABLES
        # ====================================================

        self.variables = {

            "Sphere traffic":
            "WAITING",

            "Pink target":
            "WAITING",

            "Target detection":
            "WAITING",

            "Detection error":
            None,

            "Follow mode":
            "UNKNOWN",
        }


        # ====================================================
        # GAZEBO
        # ====================================================

        self.gz_node = (
            GazeboNode()
        )


        self.gz_topic = (

            f"/world/"
            f"{args.world}"
            f"/pose/info"
        )


        subscribed = (
            self.gz_node.subscribe(

                Pose_V,

                self.gz_topic,

                self.gazebo_callback,
            )
        )


        if subscribed:

            self.get_logger().info(

                "Gazebo map tracking active: "
                f"{self.gz_topic}"
            )

        else:

            self.get_logger().error(

                "Could not subscribe to "
                "Gazebo poses."
            )


        # ====================================================
        # SPHERE CONTROL PUBLISHERS
        # ====================================================

        self.traffic_start_pub = (
            self.gz_node.advertise(

                "/sphere_network/start_traffic",

                Boolean,
            )
        )


        self.pink_start_pub = (
            self.gz_node.advertise(

                "/sphere_network/start_pink",

                Boolean,
            )
        )


        self.delete_spheres_pub = (
            self.gz_node.advertise(

                "/sphere_network/delete_all",

                Boolean,
            )
        )


        self.get_logger().info(
            "Sphere control publishers ready."
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
        # FIGURE
        # ====================================================

        self.fig = plt.figure(

            figsize=(
                12,
                8
            )
        )


        layout = (
            self.fig.add_gridspec(

                1,

                2,

                width_ratios=[
                    4,
                    1.3,
                ],
            )
        )


        self.ax = (
            self.fig.add_subplot(

                layout[
                    0,
                    0
                ]
            )
        )


        self.info_ax = (
            self.fig.add_subplot(

                layout[
                    0,
                    1
                ]
            )
        )


        # ====================================================
        # BACKGROUND IMAGE
        # ====================================================

        if os.path.exists(
            self.args.map_image
        ):

            background = (
                plt.imread(
                    self.args.map_image
                )
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

                "Map image not found: "
                f"{self.args.map_image}"
            )


        # ====================================================
        # FIXED AXES
        # ====================================================

        self.ax.set_xlim(

            self.args.xmin,

            self.args.xmax,
        )


        self.ax.set_ylim(

            self.args.ymin,

            self.args.ymax,
        )


        self.ax.set_autoscale_on(
            False
        )


        self.ax.set_aspect(

            "equal",

            adjustable="box",
        )


        self.ax.set_title(

            "UAV Project - "
            "Top Down Tracking Map"
        )


        self.ax.set_xlabel(
            "World X (m)"
        )


        self.ax.set_ylabel(
            "World Y (m)"
        )


        self.ax.grid(

            True,

            alpha=0.25,
        )


        self.info_ax.axis(
            "off"
        )


        # ====================================================
        # MARKERS
        # ====================================================

        self.points = {}

        self.trails = {}


        for (
            name,
            entity
        ) in self.entities.items():

            self.points[name] = (

                self.ax.scatter(

                    [],

                    [],

                    s=entity.size,

                    marker=entity.marker,

                    color=entity.color,

                    label=entity.label,

                    zorder=6,
                )
            )


            line, = (
                self.ax.plot(

                    [],

                    [],

                    color=entity.color,

                    linewidth=1.2,

                    alpha=0.55,

                    zorder=4,
                )
            )


            self.trails[name] = (
                line
            )


        # ====================================================
        # DRONE HEADING
        # ====================================================

        self.heading_line, = (
            self.ax.plot(

                [],

                [],

                color="tab:blue",

                linewidth=2,

                zorder=5,
            )
        )


        # ====================================================
        # TARGET VELOCITY
        # ====================================================

        self.velocity_line, = (
            self.ax.plot(

                [],

                [],

                color="tab:orange",

                linewidth=2,

                zorder=5,
            )
        )


        # ====================================================
        # DETECTION ERROR
        # ====================================================

        self.error_line, = (
            self.ax.plot(

                [],

                [],

                linestyle="--",

                color="gray",

                linewidth=1.4,

                zorder=5,
            )
        )


        self.ax.legend(
            loc="upper right"
        )


        # ====================================================
        # TELEMETRY
        # ====================================================

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
        # CONTROL BUTTONS
        # ====================================================

        self.traffic_button_ax = (
            self.fig.add_axes(

                [
                    0.50,
                    0.025,
                    0.14,
                    0.055,
                ]
            )
        )


        self.pink_button_ax = (
            self.fig.add_axes(

                [
                    0.65,
                    0.025,
                    0.14,
                    0.055,
                ]
            )
        )


        self.delete_button_ax = (
            self.fig.add_axes(

                [
                    0.80,
                    0.025,
                    0.16,
                    0.055,
                ]
            )
        )


        self.traffic_button = (
            Button(

                self.traffic_button_ax,

                "Spawn Traffic"
            )
        )


        self.pink_button = (
            Button(

                self.pink_button_ax,

                "Spawn Pink"
            )
        )


        self.delete_button = (
            Button(

                self.delete_button_ax,

                "Delete / Reset"
            )
        )


        self.traffic_button.on_clicked(
            self.spawn_traffic
        )


        self.pink_button.on_clicked(
            self.spawn_pink
        )


        self.delete_button.on_clicked(
            self.delete_all_spheres
        )


        self.fig.subplots_adjust(

            left=0.07,

            right=0.96,

            bottom=0.12,

            top=0.93,

            wspace=0.12,
        )


        # ====================================================
        # ANIMATION
        # ====================================================

        self.animation = (
            FuncAnimation(

                self.fig,

                self.update_map,

                interval=100,

                cache_frame_data=False,
            )
        )


    # ========================================================
    # SEND SPHERE COMMAND
    # ========================================================

    def send_sphere_command(
        self,
        publisher
    ):

        msg = Boolean()

        msg.data = True


        result = (
            publisher.publish(
                msg
            )
        )


        if result is False:

            return False


        return True


    # ========================================================
    # SPAWN TRAFFIC
    # ========================================================

    def spawn_traffic(
        self,
        event
    ):

        if self.send_sphere_command(
            self.traffic_start_pub
        ):

            self.variables[
                "Sphere traffic"
            ] = "RUNNING"


            self.get_logger().info(
                "Traffic start command sent."
            )

        else:

            self.variables[
                "Sphere traffic"
            ] = "SEND FAILED"


            self.get_logger().warning(
                "Traffic start command failed."
            )


    # ========================================================
    # SPAWN PINK
    # ========================================================

    def spawn_pink(
        self,
        event
    ):

        if self.send_sphere_command(
            self.pink_start_pub
        ):

            self.variables[
                "Pink target"
            ] = "ACTIVE"


            self.get_logger().info(
                "Pink target command sent."
            )

        else:

            self.variables[
                "Pink target"
            ] = "SEND FAILED"


            self.get_logger().warning(
                "Pink target command failed."
            )


    # ========================================================
    # DELETE / RESET
    # ========================================================

    def delete_all_spheres(
        self,
        event
    ):

        if not self.send_sphere_command(
            self.delete_spheres_pub
        ):

            self.get_logger().warning(
                "Delete/reset command failed."
            )

            return


        self.variables[
            "Sphere traffic"
        ] = "WAITING"


        self.variables[
            "Pink target"
        ] = "WAITING"


        self.variables[
            "Target detection"
        ] = "WAITING"


        self.variables[
            "Detection error"
        ] = None


        # Clear true and detected target
        # markers and trails.

        for name in [

            "target_true",

            "target_detected",

        ]:

            entity = (
                self.entities[
                    name
                ]
            )


            entity.x = None

            entity.y = None

            entity.z = None

            entity.vx = None

            entity.vy = None

            entity.updated_at = 0.0

            entity.trail.clear()


            self.points[
                name
            ].set_offsets(

                [[
                    float("nan"),

                    float("nan"),
                ]]
            )


            self.trails[
                name
            ].set_data(
                [],
                []
            )


        self.velocity_line.set_data(
            [],
            []
        )


        self.error_line.set_data(
            [],
            []
        )


        self.get_logger().info(
            "Delete/reset command sent."
        )


    # ========================================================
    # QUATERNION -> YAW
    # ========================================================

    def quaternion_yaw(
        self,
        q
    ):

        return math.atan2(

            2.0
            *
            (
                q.w
                *
                q.z

                +

                q.x
                *
                q.y
            ),

            1.0

            -

            2.0
            *
            (
                q.y
                *
                q.y

                +

                q.z
                *
                q.z
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
            self.entities[
                name
            ]
        )


        entity.x = float(
            x
        )


        entity.y = float(
            y
        )


        if z is not None:

            entity.z = float(
                z
            )


        if yaw is not None:

            entity.yaw = float(
                yaw
            )


        if vx is not None:

            entity.vx = float(
                vx
            )


        if vy is not None:

            entity.vy = float(
                vy
            )


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
    # ========================================================

    def gazebo_callback(
        self,
        msg
    ):

        with self.lock:

            for pose in msg.pose:

                name = (
                    pose.name
                    .split("::")[-1]
                )


                # ----------------------------------------
                # DRONE
                # ----------------------------------------

                if (
                    name
                    ==
                    "x500_gimbal_0"
                ):

                    self.update_entity(

                        "drone",

                        pose.position.x,

                        pose.position.y,

                        pose.position.z,

                        yaw=
                        self.quaternion_yaw(
                            pose.orientation
                        ),
                    )


                # ----------------------------------------
                # PINK TRUE TARGET
                # ----------------------------------------

                elif (
                    "pink"
                    in
                    name.lower()
                ):

                    self.update_entity(

                        "target_true",

                        pose.position.x,

                        pose.position.y,

                        pose.position.z,
                    )


    # ========================================================
    # TARGET ESTIMATOR CALLBACK
    # ========================================================

    def target_callback(
        self,
        msg
    ):

        try:

            data = (
                json.loads(
                    msg.data
                )
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

                data.get(
                    "z"
                ),

                vx=
                data.get(
                    "vx"
                ),

                vy=
                data.get(
                    "vy"
                ),
            )


            self.variables[
                "Target detection"
            ] = "LIVE"


    # ========================================================
    # UPDATE MAP
    # ========================================================

    def update_map(
        self,
        frame
    ):

        rclpy.spin_once(

            self,

            timeout_sec=0.0,
        )


        with self.lock:

            # ============================================
            # MARKERS / TRAILS
            # ============================================

            for (
                name,
                entity
            ) in self.entities.items():

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
                self.entities[
                    "drone"
                ]
            )


            if (
                drone.x is not None
                and
                drone.y is not None
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

            else:

                self.heading_line.set_data(
                    [],
                    []
                )


            # ============================================
            # TARGET VELOCITY
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

                target_detected.y
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

            else:

                self.velocity_line.set_data(
                    [],
                    []
                )


            # ============================================
            # DETECTION ERROR
            # ============================================

            target_true = (
                self.entities[
                    "target_true"
                ]
            )


            if (
                target_true.x
                is not None

                and

                target_true.y
                is not None

                and

                target_detected.x
                is not None

                and

                target_detected.y
                is not None
            ):

                error = (
                    math.hypot(

                        target_true.x
                        -
                        target_detected.x,

                        target_true.y
                        -
                        target_detected.y,
                    )
                )


                self.variables[
                    "Detection error"
                ] = error


                self.error_line.set_data(

                    [
                        target_true.x,

                        target_detected.x,
                    ],

                    [
                        target_true.y,

                        target_detected.y,
                    ],
                )

            else:

                self.error_line.set_data(
                    [],
                    []
                )


            # ============================================
            # TELEMETRY
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
                    self.entities[
                        name
                    ]
                )


                lines.append(
                    entity.label
                )


                if (
                    entity.x is None
                ):

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

                        and

                        entity.vy
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


                lines.append(
                    ""
                )


            lines.append(
                "VARIABLES"
            )


            lines.append(
                "========="
            )


            for (
                name,
                value
            ) in self.variables.items():

                if isinstance(
                    value,
                    float
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

        default=
        "city_test",
    )


    parser.add_argument(

        "--map-image",

        default=
        os.path.expanduser(

            "~/Documents/"
            "Simulation_City/"
            "capstone_runtime/"
            "map_system/"
            "city_topdown.png"
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

        plt.show()

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()


        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()
