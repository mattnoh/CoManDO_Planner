#!/usr/bin/env python3
"""
Diagnostic script to check what's actually being published on Crazyflie topics
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import LogDataGeneric
import sys

class TopicDiagnostic(Node):
    def __init__(self, drone_name='cf_1'):
        super().__init__('topic_diagnostic')
        
        self.drone_name = drone_name
        self.pose_count = 0
        self.vel_count = 0
        self.ang_vel_count = 0
        
        # Subscribe to all three topics
        self.pose_sub = self.create_subscription(
            PoseStamped,
            f'/{drone_name}/pose',
            self.pose_callback,
            10)
        
        self.vel_sub = self.create_subscription(
            LogDataGeneric,
            f'/{drone_name}/velocity',
            self.vel_callback,
            10)
        
        self.ang_vel_sub = self.create_subscription(
            LogDataGeneric,
            f'/{drone_name}/angular_velocity',
            self.ang_vel_callback,
            10)
        
        self.timer = self.create_timer(2.0, self.print_status)
        
        self.get_logger().info(f'Listening to topics for {drone_name}...')
        self.get_logger().info(f'  - /{drone_name}/pose')
        self.get_logger().info(f'  - /{drone_name}/velocity')
        self.get_logger().info(f'  - /{drone_name}/angular_velocity')
    
    def pose_callback(self, msg):
        self.pose_count += 1
        if self.pose_count == 1:
            self.get_logger().info(f'POSE received: pos=[{msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]')
    
    def vel_callback(self, msg):
        self.vel_count += 1
        if self.vel_count == 1:
            self.get_logger().info(f'VELOCITY received: values={msg.values}, timestamp={msg.timestamp}')
            self.get_logger().info(f'  Number of values: {len(msg.values)}')
    
    def ang_vel_callback(self, msg):
        self.ang_vel_count += 1
        if self.ang_vel_count == 1:
            self.get_logger().info(f'ANGULAR_VELOCITY received: values={msg.values}, timestamp={msg.timestamp}')
            self.get_logger().info(f'  Number of values: {len(msg.values)}')
    
    def print_status(self):
        self.get_logger().info(f'Status: pose={self.pose_count}, vel={self.vel_count}, ang_vel={self.ang_vel_count}')

def main(args=None):
    rclpy.init(args=args)
    
    drone_name = 'cf_1'
    if len(sys.argv) > 1:
        drone_name = sys.argv[1]
    
    node = TopicDiagnostic(drone_name)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()