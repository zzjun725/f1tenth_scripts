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

home = '/sim_ws/src'
# odom_topic = 'ego_racecar/odom'
# drive_topic = '/drive'
odom_topic = 'opp_racecar/odom'
drive_topic = '/opp_drive'
os.makedirs(home+'/wp_log', exist_ok = True)
log_position = home+'/wp_log'
wp_path = log_position
wp_gap = 1
node_publish = False
node_execute = False
use_optimal = True
velocity_scale = 0.5

def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length) 

class TrackingPlanner:
    def __init__(self, wp_path=wp_path, wp_gap = 0, debug=True):
        # wp
        self.wp = []
        self.wpNum = None
        self.wp_path = wp_path
        self.wpGapCounter = 0
        self.wpGapThres = wp_gap
        self.max_speed = 20
        self.speedScale = velocity_scale


        # PID for speed
        self.Increase_P = 1 / 5
        self.Decrease_P = 1 / 6
        self.P = 10
        self.targetSpeed = 0

    
    def load_wp(self):
        for file in os.listdir(wp_path):
            if file.startswith('interp'):
                wp_log = csv.reader(open(os.path.join(wp_path, file)))
                print(f'load wp_log: {file}')
                break

        points = [] # (x, y, theta)
        for i, row in enumerate(wp_log):
            if (i % wp_gap) == 0:
                points.append([float(row[0]), float(row[1]), float(row[2])])
        self.wp = np.array(points).T
        print(f'shape of wp: {self.wp.shape}')

    
    def load_Optimalwp(self): 
        for file in os.listdir(wp_path):
            if file.startswith('optimal'):
                wp_log = csv.reader(open(os.path.join(wp_path, file)))
                print('load Optimalwp_log')
                break
        for i, row in enumerate(wp_log):
            if i > 2:
                # import ipdb; ipdb.set_trace()
                if self.wpGapCounter == self.wpGapThres:
                    self.wpGapCounter = 0
                    row = row[0].split(';')
                    x, y, v = float(row[1]), float(row[2]), np.clip(float(row[5])*self.speedScale, 0, self.max_speed)
                    self.wp.append([x, y, v])
                    # TODO: fix this if the road logger is correct
                    # self.wp.append([y, x, v])
                else:
                    self.wpGapCounter += 1
        self.wp = np.array(self.wp[:-1]).T  # (3, n), n is the number of waypoints
        self.wpNum = len(self.wp[0])
        print(self.wpNum)
    
    def planning(self, pose, speed):
        """
        pose: (global_x, global_y, yaw) of the car
        speed: current speed of the car

        Return:
        steering_angle, accelation
        """
        raise NotImplementedError
        # return steering_angle, accelation


class PurePursuitPlanner(TrackingPlanner):
    def __init__(self, debug=True):
        super().__init__(wp_path=wp_path, wp_gap=0, debug=debug)
        # self.wp = []
        self.minL = 0.5
        self.maxL = 2.0
        self.minP = 0.6
        self.maxP = 0.9
        self.interpScale = 20
        self.Pscale = 5
        self.Lscale = 5
        self.interp_P_scale = (self.maxP-self.minP) / self.Pscale
        self.interp_L_scale = (self.maxL-self.minL) / self.Lscale
        self.prev_error = 0
        self.D = 0.05
        self.errthres = 0.1
        # self.load_Optimalwp()  # (3, n), n is the number of waypoints
        if use_optimal:
            self.load_Optimalwp()
        else:
            self.load_wp()
    
    def find_targetWp(self, pose, speed):
        """
        cur_positon: (2, )
        return: cur_L, targetWp(2, ), targetV 
        """

        #### get current position and transformation matrix ### 
        cur_position = pose[:2]
        targetL = speed * self.interp_L_scale + self.minL
        cur_P = self.maxP - speed * self.interp_P_scale 
        
        yaw = pose[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, pose[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, pose[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        global2local = np.linalg.inv(local2global)
        #### get current position and transformation matrix ### 

        #### get nearest waypoint ### 
        wp_xyaxis = self.wp[:2].T  # (n, 2)
        dist = np.linalg.norm(wp_xyaxis-cur_position.reshape(1, 2), axis=1)
        nearst_idx = np.argmin(dist)
        nearst_point = wp_xyaxis[nearst_idx]
        segment_end = nearst_idx
        find_p = False
        for i, point in enumerate(wp_xyaxis[nearst_idx:]):
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist > targetL:
                find_p = True
                break
        if not find_p:
            # import ipdb; ipdb.set_trace()
            for i, point in enumerate(wp_xyaxis):
                cur_dist = np.linalg.norm(cur_position-point)
                if cur_dist > targetL:
                    segment_end = i
                    break           
        else:
            segment_end += i
        target_global = wp_xyaxis[segment_end]
        target_v = self.wp[2][segment_end]
        #### get nearest waypoint ### 

        #### interpolation ### 
        segment_begin = safe_changeIdx(self.wpNum, segment_end, -1)
        x_array = np.linspace(wp_xyaxis[segment_begin][0], wp_xyaxis[segment_end][0], self.interpScale)
        y_array = np.linspace(wp_xyaxis[segment_begin][1], wp_xyaxis[segment_end][1], self.interpScale)
        v_array = np.linspace(self.wp[2][segment_begin], self.wp[2][segment_end], self.interpScale)
        xy_interp = np.vstack([x_array, y_array])
        dist_interp = np.linalg.norm(xy_interp-cur_position.reshape(2, 1), axis=0) - targetL
        i_interp = np.argmin(np.abs(dist_interp))
        target_global = np.array([x_array[i_interp], y_array[i_interp]])
        target_v = v_array[i_interp]
        #### interpolation ### 

        cur_L = np.linalg.norm(cur_position-target_global)
        # ipdb.set_trace()
        # print(segment_end)
        target_local = global2local @ np.array([target_global[0], target_global[1], 0, 1]) 
        cur_error = target_local[1]
        return cur_L, cur_P, cur_error, target_local, target_v, target_global
    
    def planning(self, cur_L, cur_P, cur_error, target_local, target_v, time_interval=None):
        """
        pose: (global_x, global_y, yaw) of the car
        """

        if time_interval:
            offset = self.D * (cur_error - self.prev_error) / time_interval
        else:
            offset = self.D * (cur_error - self.prev_error)
        self.prev_error = cur_error
        # print(f'D_offset: {offset}')

        gamma = 2*abs(target_local[1]) / (cur_L ** 2)
        if target_local[1] < 0:
            steering_angle = cur_P * -gamma
        else:
            steering_angle = cur_P * gamma
        steering_angle = np.clip(steering_angle+offset, -1.0, 1.0)

        return steering_angle, target_v


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
        self.planner = PurePursuitPlanner()
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self.pose_callback, 10)
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord)  
        self.prev_time = 0
        self.time_thres = 0.02
    
    def pose_callback(self, pose_msg):
        
        ###### publish waypoints
        if node_publish:
            wp = self.planner.wp.T # (n, 3)
            scale_vector = Vector3(x=0.1, y=0.1, z=0.1)
            marker = Marker(
                        type=Marker.POINTS,
                        id=0,
                        # action = Marker.ADD, 
                        pose=Pose(),
                        scale=scale_vector,
                        header=Header(frame_id='map'),
                        color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),                    
                        )
            for i, point in enumerate(wp):
                x, y = point[0], point[1]
                x, y = float(x), float(y)
                # print(x, y)
                point = Point(x=x, y=y, z=0.0)
                marker.points.append(point)
            self.waypoints_markerpub.publish(marker)        
        ####### publish waypoints

        ####### get current pose and speed
        cur_position = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        quaternion = np.array([pose_msg.pose.pose.orientation.w, 
                    pose_msg.pose.pose.orientation.x, 
                    pose_msg.pose.pose.orientation.y, 
                    pose_msg.pose.pose.orientation.z])
        
        euler = transforms3d.euler.quat2euler(quaternion)
        yaw = euler[2]

        pose = np.array([cur_position[0], cur_position[1], yaw])
        x_speed = pose_msg.twist.twist.linear.x
        y_speed = pose_msg.twist.twist.linear.y
        speed = np.sqrt(x_speed**2 + y_speed**2)
        ####### get current pose and speed

        ####### Planning
        cur_time = pose_msg.header.stamp.nanosec/1e9 + pose_msg.header.stamp.sec
        time_interval = cur_time - self.prev_time
        # print(time_interval)
        if self.prev_time != 0.0:
            if time_interval > self.time_thres:
                cur_L, cur_P, cur_error, target_local, target_v, target_global = self.planner.find_targetWp(pose, speed)
                steering_angle, velocity = self.planner.planning(cur_L, cur_P, cur_error, target_local, target_v, time_interval)
                self.prev_time = cur_time
            else:
                return 
        else:
            self.prev_time = cur_time
            return 
        ####### Planning

        ####### publish current target marker
        if node_publish:
            scale_vector = Vector3(x=0.2, y=0.2, z=0.2)
            marker = Marker(
                type=Marker.SPHERE,
                id=1,
                # action = Marker.ADD, 
                pose=Pose(),
                scale=scale_vector,
                header=Header(frame_id='map'),
                color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),                    
                )
            marker.pose.position = Point(x=target_global[0], y=target_global[1], z=0.0)
            self.waypoints_markerpub.publish(marker)
        ####### publish current target marker 
   
        # print(f'cur_position {cur_position}')
        # print(f'steering_angle {steering_angle}')
        # print(f'velocity {velocity}')
        # print(f'interp_point{interp_point}')
        
        ####### Sending Command
        if node_execute:
            self.ackermann_ord.drive.speed = velocity
            self.ackermann_ord.drive.steering_angle = steering_angle
            self.drive_publisher.publish(self.ackermann_ord)
        else:
            self.drive_publisher.publish(self.ackermann_ord)
        ####### Sending Command




def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
