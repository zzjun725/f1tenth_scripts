#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

import numpy as np
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import os
import csv
from nav_msgs.msg import Odometry
import transforms3d
from builtin_interfaces.msg import Duration
# TODO CHECK: include needed ROS msg type headers and libraries
import ipdb

home = '/sim_ws'
os.makedirs(home+'/wp_log', exist_ok = True)
log_position = home+'/wp_log'
for file in os.listdir(log_position):
    if file.startswith('wp'):
        wp_log = csv.reader(open(os.path.join(log_position, file)))
        # import ipdb; ipdb.set_trace()
        print('load wp_log')
waypoints = []
last_x, last_y = 0, 0
for i, row in enumerate(wp_log):
    # print(row)
    if (i % 40 != 0):
        continue
    if len(row) > 2:
        x, y = row[0], row[1]
        x, y = float(x), float(y)
        # if (abs(last_y-y) + abs(last_x-x)) > 0.5: 
        waypoints.append(np.array([x, y]))
        # last_x, last_y = x, y
wp = np.array(waypoints)
print(len(wp))


# interpolate
#      ^ x
#   y  |
# <-----
# left_up = np.array([9.56, 8.571])  # (x2, y2)
# right_down = np.array([-13.57, 0.28])  # (x1, y1)
# y1, y2 = right_down[1], left_up[1]  # y1 < y2
# x1, x2 = right_down[0], left_up[0]  # x1 < x2
# vertical = np.linspace(x1, x2, 500)
# horizon = np.linspace(y1, y2, 250)
# waypoints = []
# for v in vertical:
#     waypoints.append(np.array([v, y1]))
# for h in horizon:
#     waypoints.append(np.array([x2, h]))
# for v in vertical[::-1]:
#     waypoints.append(np.array([v, y2]))
# for h in horizon[::-1]:
#     waypoints.points
class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.waypoints_markerpub = self.create_publisher(Marker, '/wp_marker', 10)
        self.findFirstP = False
        self.nearst_idx = 0
        self.wp = None
        self.L = 0.6
        self.P = 0.3
        self.odom_subscriber = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.pose_callback, 10)
        drive_topic = '/drive'
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord)  

    def pose_callback(self, pose_msg):
        scale_vector = Vector3()
        scale_vector.x = 0.1
        scale_vector.y = 0.1
        scale_vector.z = 0.1
        marker = Marker(
                    type=Marker.POINTS,
                    id=0,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=scale_vector,
                    header=Header(frame_id='map'),
                    # color=ColorRGBA(0.0, 1.0, 0.0, 1.0),                    
                    )
        for i, point in enumerate(wp):
            x, y = point[0], point[1]
            x, y = float(x), float(y)
            # print(x, y)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            marker.points.append(point)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  
        self.waypoints_markerpub.publish(marker)        
        
        
        # TODO: find the current waypoint to track using methods mentioned in lecture
        near_dist = 100
        cur_position = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        # find nearst
        # old_nearst = self.nearst_idx
        # for i, point in enumerate(wp[self.nearst_idx:]):  # (x, y)
        #     cur_dist = np.linalg.norm(cur_position-point)
        #     if cur_dist < near_dist:
        #         near_dist = cur_dist
        #         self.nearst_idx = i
        #     if self.findFirstP:
        #         if cur_dist > self.L:
        #             break
        # self.nearst_idx += old_nearst
        # self.findFirstP = True
        # old_nearst = self.nearst_idx
        for i, point in enumerate(wp):  # (x, y)
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist < near_dist:
                near_dist = cur_dist
                self.nearst_idx = i
        # import ipdb; ipdb.set_trace()
        segment_end = self.nearst_idx
        for i, point in enumerate(wp[self.nearst_idx:]):
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist > self.L:
                break
        segment_end += i
        # get interpolation
        error = 0.01
        x_array = np.linspace(wp[segment_end-1][0], wp[segment_end][0], 10)
        y_array = np.linspace(wp[segment_end-1][1], wp[segment_end][1], 10)
        # ipdb.set_trace()
        
        interp_point = np.array([x_array[-1], y_array[-1]])
        # print(interp_point)
        for x, y in zip(x_array, y_array):
            interp_point = np.array([x, y])
            if abs(self.L - np.linalg.norm(cur_position-interp_point)) < error:
                # print('changed')
                interp_point = np.array([x, y])
        
        # print(interp_point)
        cur_L = np.linalg.norm(cur_position-interp_point)
        # TODO: transform goal point to vehicle frame of reference
        quaternion = np.array([pose_msg.pose.pose.orientation.w, 
                            pose_msg.pose.pose.orientation.x, 
                            pose_msg.pose.pose.orientation.y, 
                            pose_msg.pose.pose.orientation.z])

        euler = transforms3d.euler.quat2euler(quaternion)
        yaw = euler[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, cur_position[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, cur_position[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        
        
        
        scale_vector = Vector3()
        scale_vector.x = 0.5
        scale_vector.y = 0.5
        scale_vector.z = 0.5
        marker = Marker(
            type=Marker.SPHERE,
            id=1,
            # action = Marker.ADD, 
            pose=Pose(),
            scale=scale_vector,
            header=Header(frame_id='map'),
            # color=ColorRGBA(0.0, 1.0, 0.0, 1.0),                    
            )
        marker.pose.position.x = interp_point[0]
        marker.pose.position.y = interp_point[1]
        marker.pose.position.z = 0.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  
        self.waypoints_markerpub.publish(marker)          
        
        
        local_goalP = np.linalg.inv(local2global) @ np.array([interp_point[0], interp_point[1], 0, 1])
        print(local_goalP)
        gamma = 2*abs(local_goalP[1]) / (cur_L ** 2)
        # TODO: calculate curvature/steering angle
        if local_goalP[1] > 0:
            steering_angle = self.P * gamma
        else:
            steering_angle = self.P * -gamma
        # TODO: publish drive message, don't forget to limit the steering angle.
        velocity = 1.0
        if abs(steering_angle) >=1:
            steering_angle /= 4
        
        # print(f'cur_position {cur_position}')
        # print(f'local_goal {local_goalP}')
        # print(f'interp_point{interp_point}')
        self.ackermann_ord.drive.speed = velocity
        self.ackermann_ord.drive.steering_angle = steering_angle
        self.drive_publisher.publish(self.ackermann_ord)           


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
