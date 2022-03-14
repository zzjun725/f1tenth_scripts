#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import os
import csv
# TODO CHECK: include needed ROS msg type headers and libraries

home = '/sim_ws'

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.waypoints_markerpub = self.create_publisher(Marker, 'wp_marker', 10)
        self.drawWayPoints()

    def _pubMarker(self, x, y):
        marker = Marker(
                    type=Marker.LINE_STRIP,
                    id=0,
                    # lifetime=rospy.Duration(1.5),
                    pose=Pose(Point(x, y, 0), Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0.06, 0.06, 0.06),
                    header=Header(frame_id='base_link'),
                    color=ColorRGBA(0.0, 1.0, 0.0, 1.0),                    )
        self.waypoints_markerpub.publish(marker)        
    
    def drawWayPoints(self):
        log_position = home+'/wp_log'
        for file in os.listdir(log_position):
            if file.startswith('wp'):
                wp_log = csv.reader(os.path.join(log_position, file))
        waypoints = []
        for row in wp_log:
            waypoints.append(row)
            if len(row) > 2:
                x, y = row[0], row[1]
                print(x, y)
                self._pubMarker(x, y)


        # waypoints = np.array(waypoints)
    
    def pose_callback(self, pose_msg):
        pass
        # TODO: find the current waypoint to track using methods mentioned in lecture

        # TODO: transform goal point to vehicle frame of reference

        # TODO: calculate curvature/steering angle

        # TODO: publish drive message, don't forget to limit the steering angle.
    


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
