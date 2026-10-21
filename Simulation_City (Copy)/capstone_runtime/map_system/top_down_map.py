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
from matplotlib.widgets import Button, Slider

import rclpy

from rclpy.node import Node

from std_msgs.msg import String

try:
    from gz.transport13 import Node as GazeboNode
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.double_pb2 import Double

except ImportError:
    from gz.transport14 import Node as GazeboNode
    from gz.msgs11.pose_v_pb2 import Pose_V
    from gz.msgs11.boolean_pb2 import Boolean
    from gz.msgs11.double_pb2 import Double

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
        default_factory=lambda: deque(maxlen=200)
    )

class TopDownMap(Node):

    def __init__(self, args):

        super().__init__("top_down_map")

        self.args = args
        self.lock = threading.Lock()

        self.last_gazebo_process_time = 0.0
        self.gazebo_process_period = 0.1

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

        self.variables = {
            "Sphere traffic": "WAITING",
            "Pink target": "WAITING",
            "Pink speed": "6.0 m/s",
            "Target detection": "WAITING",
            "Detection error": None,
            "Follow mode": "UNKNOWN",
        }

        # GAZEBO

        self.gz_node = GazeboNode()

        self.gz_topic = (
            f"/world/{args.world}/pose/info"
        )
        subscribed = self.gz_node.subscribe(
            Pose_V,
            self.gz_topic,
            self.gazebo_callback,
        )
        if subscribed:
            self.get_logger().info(
                f"Gazebo map tracking active: {self.gz_topic}"
            )
        else:
            self.get_logger().error(
                "Could not subscribe to Gazebo poses."
            )

        # SPHERE CONTROL

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
        self.pink_speed_pub = (
            self.gz_node.advertise(
                "/sphere_network/pink_speed",
                Double,
            )
        )

        # TARGET ESTIMATOR

        self.create_subscription(
            String,
            "/target_estimator/target",
            self.target_callback,
            10,
        )

        # FIGURE

        self.fig = plt.figure(
            figsize=(12, 8)
        )
        layout = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[4, 1.3],
        )
        self.ax = self.fig.add_subplot(
            layout[0, 0]
        )
        self.info_ax = self.fig.add_subplot(
            layout[0, 1]
        )

        # BACKGROUND IMAGE

        if os.path.exists(
            self.args.map_image
        ):
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

        # FIXED AXES

        self.ax.set_xlim(
            self.args.xmin,
            self.args.xmax,
        )
        self.ax.set_ylim(
            self.args.ymin,
            self.args.ymax,
        )
        self.ax.set_autoscale_on(False)
        self.ax.set_aspect(
            "equal",
            adjustable="box",
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
        self.ax.grid(
            True,
            alpha=0.25,
        )
        self.info_ax.axis(
            "off"
        )

        # MARKERS

        self.points = {}
        self.trails = {}

        for name, entity in self.entities.items():

            self.points[name] = self.ax.scatter(
                [],
                [],
                s=entity.size,
                marker=entity.marker,
                color=entity.color,
                label=entity.label,
                zorder=6,
            )

            line, = self.ax.plot(
                [],
                [],
                color=entity.color,
                linewidth=1.2,
                alpha=0.55,
                zorder=4,
            )

            self.trails[name] = line

        self.heading_line, = self.ax.plot(
            [],
            [],
            color="tab:blue",
            linewidth=2,
            zorder=5,
        )
        self.velocity_line, = self.ax.plot(
            [],
            [],
            color="tab:orange",
            linewidth=2,
            zorder=5,
        )
        self.error_line, = self.ax.plot(
            [],
            [],
            linestyle="--",
            color="gray",
            linewidth=1.4,
            zorder=5,
        )
        self.ax.legend(
            loc="upper right"
        )
        self.info_text = self.info_ax.text(
            0.02,
            0.98,
            "Waiting for data...",
            va="top",
            family="monospace",
            fontsize=10,
        )

        # BUTTONS

        self.traffic_button_ax = self.fig.add_axes(
            [
                0.50,
                0.025,
                0.14,
                0.055,
            ]
        )
        self.pink_button_ax = self.fig.add_axes(
            [
                0.65,
                0.025,
                0.14,
                0.055,
            ]
        )
        self.delete_button_ax = self.fig.add_axes(
            [
                0.80,
                0.025,
                0.16,
                0.055,
            ]
        )
        self.traffic_button = Button(
            self.traffic_button_ax,
            "Spawn Traffic"
        )
        self.pink_button = Button(
            self.pink_button_ax,
            "Spawn Pink"
        )
        self.delete_button = Button(
            self.delete_button_ax,
            "Delete / Reset"
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

        # PINK SPEED SLIDER

        self.pink_speed_ax = self.fig.add_axes(
            [
                0.50,
                0.09,
                0.43,
                0.03,
            ]
        )
        self.pink_speed_slider = Slider(
            self.pink_speed_ax,
            "Pink Speed",
            0.0,
            15.0,
            valinit=6.0,
            valstep=0.5,
            valfmt="%.1f m/s",
        )
        self.pink_speed_slider.on_changed(
            self.set_pink_speed
        )
        self.fig.subplots_adjust(
            left=0.07,
            right=0.96,
            bottom=0.16,
            top=0.93,
            wspace=0.12,
        )
        self.animation = FuncAnimation(
            self.fig,
            self.update_map,
            interval=100,
            cache_frame_data=False,
        )
    def send_sphere_command(
        self,
        publisher
    ):

        msg = Boolean()
        msg.data = True

        result = publisher.publish(
            msg
        )

        return result is not False

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

    def set_pink_speed(
        self,
        speed
    ):

        msg = Double()
        msg.data = float(speed)

        self.pink_speed_pub.publish(
            msg
        )

        self.variables[
            "Pink speed"
        ] = f"{float(speed):.1f} m/s"

    def delete_all_spheres(
        self,
        event
    ):

        if not self.send_sphere_command(
            self.delete_spheres_pub
        ):
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

        for name in [
            "target_true",
            "target_detected",
        ]:

            entity = self.entities[
                name
            ]

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

    def quaternion_yaw(
        self,
        q
    ):

        return math.atan2(

            2.0 * (
                q.w * q.z
                +
                q.x * q.y
            ),

            1.0
            -
            2.0 * (
                q.y * q.y
                +
                q.z * q.z
            ),
        )

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

        entity = self.entities[
            name
        ]

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

    # OPTIMISED GAZEBO CALLBACK

    def gazebo_callback(
        self,
        msg
    ):

        now = time.monotonic()

        if (
            now
            -
            self.last_gazebo_process_time
            <
            self.gazebo_process_period
        ):
            return

        self.last_gazebo_process_time = (
            now
        )

        drone_pose = None
        pink_pose = None

        for pose in msg.pose:

            name = pose.name

            if (
                name == "x500_gimbal_0"
                or
                name.endswith(
                    "::x500_gimbal_0"
                )
            ):

                drone_pose = pose

            elif (
                name == "pink_route_sphere"
                or
                name.endswith(
                    "::pink_route_sphere"
                )
            ):

                pink_pose = pose

            if (
                drone_pose is not None
                and
                pink_pose is not None
            ):

                break

        with self.lock:

            if drone_pose is not None:

                self.update_entity(

                    "drone",

                    drone_pose.position.x,

                    drone_pose.position.y,

                    drone_pose.position.z,

                    yaw=
                    self.quaternion_yaw(
                        drone_pose.orientation
                    ),
                )

            if pink_pose is not None:

                self.update_entity(

                    "target_true",

                    pink_pose.position.x,

                    pink_pose.position.y,

                    pink_pose.position.z,
                )

    def target_callback(
        self,
        msg
    ):

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
                "Target detection"
            ] = "LIVE"

    def update_map(
        self,
        frame
    ):

        rclpy.spin_once(
            self,
            timeout_sec=0.0,
        )

        with self.lock:

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
                        for p in entity.trail
                    ]

                    ys = [
                        p[1]
                        for p in entity.trail
                    ]

                    self.trails[
                        name
                    ].set_data(
                        xs,
                        ys,
                    )

            drone = self.entities[
                "drone"
            ]

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

            target_detected = self.entities[
                "target_detected"
            ]

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

            target_true = self.entities[
                "target_true"
            ]

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

                error = math.hypot(

                    target_true.x
                    -
                    target_detected.x,

                    target_true.y
                    -
                    target_detected.y,
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

                entity = self.entities[
                    name
                ]

                lines.append(
                    entity.label
                )

                if entity.x is None:

                    lines.append(
                        "  waiting..."
                    )

                else:

                    lines.append(
                        f"  X: {entity.x:8.2f} m"
                    )

                    lines.append(
                        f"  Y: {entity.y:8.2f} m"
                    )

                    if entity.z is not None:

                        lines.append(
                            f"  Z: {entity.z:8.2f} m"
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
                    f"{name}: {value}"
                )

            self.info_text.set_text(

                "\n".join(
                    lines
                )
            )

        return []


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--world",
        default="city_test",
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

    args = parser.parse_args()

    rclpy.init()

    node = TopDownMap(
        args
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
