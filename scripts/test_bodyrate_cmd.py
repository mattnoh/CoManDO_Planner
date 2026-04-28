#!/usr/bin/env python3
"""test_bodyrate_cmd.py — publishes a constant cmd_bodyrate at 50 Hz for a configurable duration.

Used to verify the crazyswarm2 cmd_bodyrate patch end-to-end without the full planner stack.

Parameters:
  drone_name  (str)   ROS namespace           default: cf_1
  thrust      (float) specific thrust [m/s²]  default: 9.81  (hover)
  wx          (float) roll rate   [rad/s]      default: 0.0
  wy          (float) pitch rate  [rad/s]      default: 0.0
  wz          (float) yaw rate    [rad/s]      default: 0.0
  duration    (float) seconds to publish       default: 5.0

Usage:
  ros2 run comando_planner test_bodyrate_cmd.py --ros-args -p drone_name:=cf_1
  ros2 run comando_planner test_bodyrate_cmd.py --ros-args \\
      -p drone_name:=cf_1 -p thrust:=9.81 -p wz:=0.52 -p duration:=3.0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class TestBodyrateCmd(Node):
    def __init__(self):
        super().__init__("test_bodyrate_cmd")

        self.declare_parameter("drone_name", "cf_1")
        self.declare_parameter("thrust", 17.81)
        self.declare_parameter("wx", 0.0)
        self.declare_parameter("wy", 0.0)
        self.declare_parameter("wz", 0.0)
        self.declare_parameter("duration", 20.0)

        drone_name = self.get_parameter("drone_name").get_parameter_value().string_value
        self._thrust = self.get_parameter("thrust").get_parameter_value().double_value
        self._wx = self.get_parameter("wx").get_parameter_value().double_value
        self._wy = self.get_parameter("wy").get_parameter_value().double_value
        self._wz = self.get_parameter("wz").get_parameter_value().double_value
        self._duration = self.get_parameter("duration").get_parameter_value().double_value

        topic = f"/{drone_name}/cmd_bodyrate"
        self._pub = self.create_publisher(TwistStamped, topic, 10)
        self._start = self.get_clock().now()
        self._timer = self.create_timer(0.02, self._tick)  # 50 Hz

        self.get_logger().info(
            f"[TestBodyrateCmd] publishing to {topic} for {self._duration}s  "
            f"T={self._thrust} m/s²  wx={self._wx} wy={self._wy} wz={self._wz} rad/s"
        )

    def _tick(self):
        elapsed = (self.get_clock().now() - self._start).nanoseconds * 1e-9
        if elapsed >= self._duration:
            self.get_logger().info("[TestBodyrateCmd] duration elapsed, stopping")
            self._timer.cancel()
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.z = self._thrust
        msg.twist.angular.x = self._wx
        msg.twist.angular.y = self._wy
        msg.twist.angular.z = self._wz
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestBodyrateCmd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
