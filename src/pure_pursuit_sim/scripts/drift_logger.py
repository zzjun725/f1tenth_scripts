#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from time import gmtime, strftime
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
# TODO CHECK: include needed ROS msg type headers and libraries
# from tf_transformations import euler_from_quaternion
import transforms3d
import os
from math import atan2, pi, cos, sin

home = os.path.dirname(__file__)
os.makedirs(home+'/wp_log', exist_ok = True)

class DriftsLogger(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('WaypointsLogger')
        lidarscan_topic = '/scan'
        self.log = open(home+'/wp_log/drift' +'.csv', 'w')
        self.scan_subscriber = self.create_subscription(
            LaserScan, lidarscan_topic, self.logger_callback, 10
        )
        # self.times = 0

    def logger_callback(self, scan_msg):
        # if self.times <= 2:
        #     self.times += 1
        
        # else:
        angle_increment = scan_msg.angle_increment
        angle_min = scan_msg.angle_min
        distances = scan_msg.ranges 

        theta = pi/6
        left_angle = pi/2
        b = distances[int((left_angle - angle_min) // angle_increment)]
        a = distances[int((left_angle + theta - angle_min) // angle_increment)]
        alpha = atan2(a*cos(theta)-b, a*sin(theta))
        left_D = b*cos(alpha)

        right_angle = -pi/2
        b = distances[int((right_angle - angle_min) // angle_increment)]
        a = distances[int((right_angle + theta - angle_min) // angle_increment)]
        alpha = atan2(a*cos(theta)-b, a*sin(theta))
        right_D = b*cos(alpha)


        self.log.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                        data.pose.pose.position.y,
                                        euler[2],
                                        speed))

def main(args=None):
    rclpy.init(args=args)
    print("Waypoints Logger Initialized")
    waypoints_logger = DriftsLogger()
    rclpy.spin(waypoints_logger)

    waypoints_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()