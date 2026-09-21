#!/usr/bin/env python3

import math
import threading

from gz.transport13 import Node as GazeboNode
from gz.msgs10.pose_v_pb2 import Pose_V

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# PX4 ROS2 interface messages (Nodes)
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleAttitude,
)

# Used to detect held/released key presses (manual control)
from pynput import keyboard

from std_msgs.msg import Float64

class KeyboardOffboard(Node):

    def publish_zoom(self):
        zoom = self.zoom_levels[self.zoom_index]

        msg = Float64()
        msg.data = float(zoom)

        self.zoom_pub.publish(msg)

        self.get_logger().info(
            f'Camera zoom: {zoom:.1f}x'
        )

    # =========================================================
    # GAZEBO POSITION CALLBACK
    # =========================================================

    def gazebo_pose_callback(self, msg):

        for pose in msg.pose:

            if (
                pose.name == self.gz_model_name
                or pose.name.endswith(
                    f'::{self.gz_model_name}'
                )
            ):

                with self.gz_position_lock:

                    self.gz_position = (
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z)
                    )

                return


    # =========================================================
    # PRINT CURRENT GAZEBO XYZ
    # =========================================================

    def print_gazebo_position(self):

        with self.gz_position_lock:

            if self.gz_position is None:

                print()
                print('Drone position not available yet.')
                print()

                return

            x, y, z = self.gz_position

        print()
        print('==============================')
        print(' DRONE POSITION')
        print('==============================')
        print(f'X = {x:.3f}')
        print(f'Y = {y:.3f}')
        print(f'Z = {z:.3f}')
        print('==============================')
        print()

    def __init__(self):
        super().__init__('keyboard_offboard')

        # Gazebo model and world names for tracking the drone's position in the simulation.
        self.gz_world_name = 'city_test'
        self.gz_model_name = 'x500_gimbal_0'

        self.gz_pose_topic = (
            f'/world/{self.gz_world_name}/pose/info'
        )

        self.gz_position = None
        self.gz_position_lock = threading.Lock()

        # Prevent P from repeatedly printing while held
        self.position_key_latched = False

        self.gz_node = GazeboNode()

        subscribed = self.gz_node.subscribe(
            Pose_V,
            self.gz_pose_topic,
            self.gazebo_pose_callback
        )

        if subscribed:
            self.get_logger().info(
                f'Gazebo XYZ tracking active for {self.gz_model_name}'
            )
        else:
            self.get_logger().warning(
                f'Could not subscribe to {self.gz_pose_topic}'
            )

        # Drone control settings
        self.horizontal_speed = 10          # m/s
        self.vertical_speed = 5             # m/s
        self.yaw_rate = math.radians(90)    # rad/s

        self.yaw = 0.0

        # Stores all keys currently pressed. Used to detect held/released key presses.
        self.keys = set()

        # Lock to protect access to the keys set from multiple threads (keyboard listener and control loop)
        self.lock = threading.Lock()

        # PX4 doesn't allow offboard control until a few setpoints have been sent. This counter is used to track how many setpoints have been sent.
        self.counter = 0
        self.started = False

        # Drone gimbal settings
        self.gimbal_pitch = 0.0
        self.gimbal_yaw = 0.0

        self.gimbal_rate = math.radians(90.0)   # rad/s
        self.dt = 0.05                          # 20 Hz

        self.gimbal_pitch_min = math.radians(-135.0)
        self.gimbal_pitch_max = math.radians(45.0)

        # Camera zoom settings
        self.zoom_levels = [
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0
        ]

        self.zoom_index = 0
        self.zoom_in_latched = False
        self.zoom_out_latched = False

        # QoS profile for publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers and subscribers
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos
        )

        # Sends trajectory setpoints to the drone. The drone will follow these setpoints when in offboard mode.
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos
        )

        # Sends vehicle commands to the drone. Used to arm the drone and enter offboard mode.
        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos
        )

        # Subscribes to the vehicle attitude topic to get the current yaw of the drone. This is used to convert body-relative commands to NED coordinates.
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos
        )

        # Gimbal publishers for controlling the gimbal pitch angle
        self.gimbal_pitch_pub = self.create_publisher(
            Float64,
            '/model/x500_gimbal_0/command/gimbal_pitch_demo',
            10
        )

        # Gimbal yaw publisher for controlling the gimbal yaw angle
        self.gimbal_yaw_pub = self.create_publisher(
            Float64,
            '/model/x500_gimbal_0/command/gimbal_yaw_demo',
            10
        )

        # Camera zoom publisher. Bridged ROS -> Gazebo.
        self.zoom_pub = self.create_publisher(
            Float64,
            '/model/x500_gimbal_0/camera/zoom/cmd_zoom',
            10
        )

        # Start the keyboard listener in a separate thread to detect key presses and releases
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

        self.listener.start()

        # 20 Hz control loop
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            '\n'
            '===============================\n'
            ' PX4 KEYBOARD CONTROL\n'
            '===============================\n'
            ' W / S     Forward / Backward\n'
            ' A / D     Left / Right\n'
            ' Q / E     Yaw Left / Right\n'
            ' SHIFT     Up\n'
            ' CTRL      Down\n'
            ' X         Stop\n'
            '\n'
            ' UP / DOWN   Gimbal Tilt\n'
            ' LEFT/RIGHT  Gimbal Pan\n'
            ' P           Print Gazebo XYZ\n'
            ' [           Zoom Out\n'
            ' ]           Zoom In\n'
            '==============================='
        )


    def timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    # ------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------

    def normalise_key(self, key):
        # Convert key to lowercase if it's a character, otherwise return the key as is.
        try:
            return key.char.lower()
        except AttributeError:
            return key

    def on_press(self, key):
        normalised = self.normalise_key(key)

        if normalised == 'p':
            if not self.position_key_latched:
                self.position_key_latched = True
                self.print_gazebo_position()
            return

        if normalised == ']':
            if not self.zoom_in_latched:
                self.zoom_in_latched = True

                if self.zoom_index < len(self.zoom_levels) - 1:
                    self.zoom_index += 1

                self.publish_zoom()
            return

        if normalised == '[':
            if not self.zoom_out_latched:
                self.zoom_out_latched = True

                if self.zoom_index > 0:
                    self.zoom_index -= 1

                self.publish_zoom()
            return

        with self.lock:
            self.keys.add(normalised)

    def on_release(self, key):
        normalised = self.normalise_key(key)

        if normalised == 'p':
            self.position_key_latched = False
            return

        if normalised == ']':
            self.zoom_in_latched = False
            return

        if normalised == '[':
            self.zoom_out_latched = False
            return

        with self.lock:
            self.keys.discard(normalised)

    def pressed(self, key):
        # Check if a key is currently pressed by checking if it is in the set of currently pressed keys.
        with self.lock:
            return key in self.keys


    # ------------------------------------------------------
    # Vehicle orientation
    # ------------------------------------------------------

    def attitude_callback(self, msg):

        # PX4 quaternion order = w, x, y, z
        w = msg.q[0]
        x = msg.q[1]
        y = msg.q[2]
        z = msg.q[3]

        # Convert quaternion to yaw angle (in radians) using the formula for yaw from a quaternion.
        self.yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )

    # ------------------------------------------------------
    # PX4 commands
    # ------------------------------------------------------

    # Publish a vehicle command to the PX4. This is used to arm the drone and enter offboard mode.
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):

        msg = VehicleCommand()

        msg.timestamp = self.timestamp()

        msg.param1 = float(param1)
        msg.param2 = float(param2)

        msg.command = command

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1

        msg.from_external = True

        self.command_pub.publish(msg)

    # Arms the drone and requests offboard mode. This is called after a few setpoints have been sent to the drone.
    def enter_offboard(self):

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0
        )

        self.get_logger().info('Offboard mode requested')

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0
        )

        self.get_logger().info('Arm command sent')

    # ------------------------------------------------------
    # Offboard heartbeat
    # ------------------------------------------------------

    def publish_offboard_mode(self):

        msg = OffboardControlMode()

        msg.timestamp = self.timestamp()

        # Set the offboard control mode to only control velocity. Position, acceleration, attitude, and body rate control are disabled.
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False

        self.offboard_pub.publish(msg)

    # ------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------

    # The main control loop runs at 20 Hz and handles keyboard input to control the drone's movement and gimbal orientation. It sends trajectory setpoints to the drone based on the keys pressed, and also manages the gimbal pitch and yaw angles.
    def control_loop(self):

        self.publish_offboard_mode()

        # Send setpoints before entering Offboard.
        # 20 cycles at 20 Hz = 1 second.
        if not self.started:
            self.counter += 1

            if self.counter >= 20:
                self.enter_offboard()
                self.started = True

        # Body-relative commands
        forward = 0.0
        right = 0.0
        down = 0.0
        yaw_speed = 0.0

        if self.pressed('w'):
            forward += self.horizontal_speed

        if self.pressed('s'):
            forward -= self.horizontal_speed

        if self.pressed('d'):
            right += self.horizontal_speed

        if self.pressed('a'):
            right -= self.horizontal_speed

        # PX4 uses NED coordinates:
        # negative Z = UP
        # positive Z = DOWN

        if (
            self.pressed(keyboard.Key.shift) or
            self.pressed(keyboard.Key.shift_l) or
            self.pressed(keyboard.Key.shift_r)
        ):
            down -= self.vertical_speed

        if (
            self.pressed(keyboard.Key.ctrl) or
            self.pressed(keyboard.Key.ctrl_l) or
            self.pressed(keyboard.Key.ctrl_r)
        ):
            down += self.vertical_speed

        # NED positive yaw = clockwise
        if self.pressed('q'):
            yaw_speed -= self.yaw_rate

        if self.pressed('e'):
            yaw_speed += self.yaw_rate

        if self.pressed('x'):
            forward = 0.0
            right = 0.0
            down = 0.0
            yaw_speed = 0.0

        # ------------------------------------------------------
        # Gimbal control
        # ------------------------------------------------------

        # Up arrow = camera up
        gimbal_changed = False

        if self.pressed(keyboard.Key.up):
            self.gimbal_pitch += self.gimbal_rate * self.dt
            gimbal_changed = True

        if self.pressed(keyboard.Key.down):
            self.gimbal_pitch -= self.gimbal_rate * self.dt
            gimbal_changed = True

        if self.pressed(keyboard.Key.left):
            self.gimbal_yaw -= self.gimbal_rate * self.dt
            gimbal_changed = True

        if self.pressed(keyboard.Key.right):
            self.gimbal_yaw += self.gimbal_rate * self.dt
            gimbal_changed = True


        self.gimbal_pitch = max(
            self.gimbal_pitch_min,
            min(self.gimbal_pitch_max, self.gimbal_pitch)
        )


        if gimbal_changed:

            pitch_msg = Float64()
            pitch_msg.data = self.gimbal_pitch
            self.gimbal_pitch_pub.publish(pitch_msg)

            yaw_msg = Float64()
            yaw_msg.data = self.gimbal_yaw
            self.gimbal_yaw_pub.publish(yaw_msg)

        # Convert body-relative movement to NED coordinates using the current yaw of the drone.
        north = (
            forward * math.cos(self.yaw)
            - right * math.sin(self.yaw)
        )

        east = (
            forward * math.sin(self.yaw)
            + right * math.cos(self.yaw)
        )

        # Send trajectory setpoint to the drone. 
        msg = TrajectorySetpoint()

        nan = float('nan')

        msg.timestamp = self.timestamp()

        # Set the position to NaN since we are only controlling velocity in offboard mode. The drone will maintain its current position.
        msg.position = [nan, nan, nan]

        # NED Velocity commands. PX4 uses NED coordinates, so positive north, east, and down velocities are used.
        msg.velocity = [
            float(north),
            float(east),
            float(down)
        ]

        msg.acceleration = [nan, nan, nan]
        msg.jerk = [nan, nan, nan]

        msg.yaw = nan
        msg.yawspeed = float(yaw_speed)

        self.trajectory_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = KeyboardOffboard()

    try:
        # Spin the node to keep it running and processing callbacks. This will keep the control loop and keyboard listener active.
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully to stop the node and exit the program.
        pass

    finally:
        # Clean up and shut down the node and ROS2.
        node.listener.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()