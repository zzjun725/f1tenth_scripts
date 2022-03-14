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

home = '/sim_ws'


class WaypointsLogger(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('WaypointsLogger')
        self.log = open(strftime(home+'/wp_log/wp-%Y-%m-%d-%H-%M-%S',gmtime())+'.csv', 'w')
        self.odom_subscriber = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.logger_callback, 10)
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


        self.log.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                        data.pose.pose.position.y,
                                        euler[2],
                                        speed))

def main(args=None):
    rclpy.init(args=args)
    print("Waypoints Logger Initialized")
    waypoints_logger = WaypointsLogger()
    rclpy.spin(waypoints_logger)

    waypoints_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()