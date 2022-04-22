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

home = '/sim_ws/src'
os.makedirs(home+'/wp_log', exist_ok = True)

class WaypointsLogger(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('WaypointsLogger')
        self.log = open(home + '/wp_log/wp_sim' +'.csv', 'w')
        self.odom_subscriber = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.logger_callback, 10)
        self.last_x = 0
        # self.times = 0
        

    def logger_callback(self, data):
        # if self.times <= 2:
        #     self.times += 1
        
        # else:
        #     self.times = 0
        quaternion = np.array([data.pose.pose.orientation.w, 
                            data.pose.pose.orientation.x, 
                            data.pose.pose.orientation.y, 
                            data.pose.pose.orientation.z])

        euler = transforms3d.euler.quat2euler(quaternion)
        speed = np.linalg.norm(np.array([data.twist.twist.linear.x, 
                                data.twist.twist.linear.y, 
                                data.twist.twist.linear.z]),2)


        x, y = data.pose.pose.position.x, data.pose.pose.position.y
        if abs(x-self.last_x) > 0.001:
            self.log.write('%f, %f, %f, %f\n' % (x,
                                            y,
                                            euler[2],
                                            speed))
            self.last_x = x

def main(args=None):
    rclpy.init(args=args)
    print("Waypoints Logger Initialized")
    waypoints_logger = WaypointsLogger()
    rclpy.spin(waypoints_logger)

    waypoints_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()