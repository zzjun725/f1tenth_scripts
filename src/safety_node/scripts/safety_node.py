#!/usr/bin/env python3
import rclpy
from math import cos, pi

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.node import Node


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        
        self.ttc_thres = 1.5
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.
        # TODO: create ROS subscribers and publishers.
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x
        # 
        # self.get_logger().info(f'Current speed {self.speed}' )
        # print(f'Current speed {self.speed}')

    def scan_callback(self, scan_msg):
        # TODO: calculate TTC
        # print('start_scan')
        distances = scan_msg.ranges[::40]
        angle_increment = scan_msg.angle_increment*40
        angle_min = scan_msg.angle_min
        for idx, distance in enumerate(distances):
            angle = angle_min+idx*angle_increment
            r_hat = self.speed*cos(angle)
            # print(f'current_angle{angle}', f'current_distance{distance}')
            if r_hat <= 1e-2 :
                ttc = self.ttc_thres + 1
            else:
                ttc = distance / r_hat
            # print(f'current_ttc:  {ttc}')
            if ttc < self.ttc_thres:
                # TODO: publish command to brake
                self.get_logger().info(f'break at ttc= {ttc}')
                self.ackermann_ord.drive.speed = 0.
                self.drive_publisher.publish(self.ackermann_ord)
                break


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    print('start node')
    rclpy.spin(safety_node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()