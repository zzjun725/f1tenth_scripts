#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import einsum_path, linalg as LA
import math
import ipdb

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid, MapMetaData
from math import pi, atan2, cos, sin
import transforms3d
from geometry_msgs.msg import Vector3, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
import os
import csv
# import matplotlib.pyplot as plt

home = '/sim_ws/src'
# pose_topic = 'opp_racecar/odom'
# drive_topic = '/opp_drive'
# scan_topic = '/opp_scan'

pose_topic = 'ego_racecar/odom'
drive_topic = '/drive'
scan_topic = '/scan'

os.makedirs(home+'/wp_log', exist_ok = True)
log_position = home+'/wp_log'
wp_path = log_position
wp_gap = 1
node_publish = True
node_execute = True
use_optimal = True
velocity_scale = 0.8

# gridWorld
OGrid_R = 0.05
OGrid_x_min = 0
OGrid_x_max = 4
OGrid_y_min = -3
OGrid_y_max = 3

default_check_span = 0.2
check_offset = 0.05
check_interp_num = 10
avoid_span = 0.3
avoid_offset = 0.2
avoid_interp_num = 5
normal_span = 2.5
normal_interpScale = 20

# 1, -pi/2, outer
# 2, pi/2, inner

def generateDir(leftRanges, rightRanges):
    left = []; right = [];
    for l in leftRanges:
        left.extend(list(range(*l)))
    for r in rightRanges:
        right.extend(list(range(*r)))
    return left, right
left_idx, right_idx = generateDir([(0, 8), (13, 25), (31, 45)], [(8, 13), (25, 31), (45, 55)])

###### Utils functions ######
def get_span_from_two_point(span_L=0.3, interp_num=10, pA=None, pB=None, return_point=False, class_num=5):
    # vector from A to B
    x_axisVector = np.array([1.0, 0.0])
    AB_Vector = pB - pA
    # AB_th = np.arccos(np.dot(AB_Vector, x_axisVector) / (LA.norm(AB_Vector, ord=2)*1.0))
    AB_th = np.arctan2(pB[1]-pA[1], pB[0]-pA[0])
    # print(np.dot(AB_Vector, x_axisVector))
    # print(LA.norm(AB_Vector, ord=2))
    # print(AB_th*180/np.pi)
    AB_interpX = np.linspace(pA[0], pB[0], num=interp_num)
    AB_interpY = np.linspace(pA[1], pB[1], num=interp_num)
    AB_interp = np.vstack([AB_interpX, AB_interpY])
    AB_span1 = np.vstack([AB_interpX + span_L*np.cos(AB_th-np.pi/2), AB_interpY + span_L*np.sin(AB_th-np.pi/2) ])
    AB_span2 = np.vstack([AB_interpX + span_L*np.cos(AB_th+np.pi/2), AB_interpY + span_L*np.sin(AB_th+np.pi/2) ])
    if return_point:
        return AB_span1[:, 0], AB_span1[:, -1], AB_span2[:, 0], AB_span2[:, -1]
    else:
        if class_num == 5:
            return np.hstack([AB_interp, AB_span1, AB_span2, (AB_interp + AB_span1)/2, (AB_interp + AB_span2)/2])
        if class_num == 9:
            check_line_1 = (AB_interp + AB_span1)/2
            check_line_2 = (AB_interp + AB_span2)/2
            return np.hstack([AB_interp, AB_span1, AB_span2, check_line_1, check_line_2,
                              (check_line_1 + AB_span1)/2, (AB_span1 + AB_interp)/2, 
                              (AB_interp + check_line_2)/2, (check_line_2 + AB_span2)/2])

def get_normal_from_two_point(normal_L=normal_span, interp_num = normal_span*normal_interpScale, pA=None, pB=None):
    x_axisVector = np.array([1.0, 0.0])
    AB_Vector = pB - pA
    AB_midpoint = (pB+pA) / 2
    # AB_th = np.arccos(np.dot(AB_Vector, x_axisVector) / (LA.norm(AB_Vector, ord=2)*1.0))
    AB_th = np.arctan2(pB[1]-pA[1], pB[0]-pA[0])
    AB_norm1 = np.array([AB_midpoint[0] + normal_L*np.cos(AB_th-np.pi/2), AB_midpoint[1] + normal_L*np.sin(AB_th-np.pi/2)])
    AB_norm2 = np.array([AB_midpoint[0] + normal_L*np.cos(AB_th+np.pi/2), AB_midpoint[1] + normal_L*np.sin(AB_th+np.pi/2)])
    norm_interp_X1 = np.linspace(AB_norm1[0], AB_midpoint[0], num=int(interp_num))
    norm_interp_Y1 = np.linspace(AB_norm1[1], AB_midpoint[1], num=int(interp_num))

    norm_interp_X2 = np.linspace(AB_norm2[0], AB_midpoint[0], num=int(interp_num))
    norm_interp_Y2 = np.linspace(AB_norm2[1], AB_midpoint[1], num=int(interp_num))
    norm_interp1 = np.vstack([norm_interp_X1, norm_interp_Y1])
    norm_interp2 = np.vstack([norm_interp_X2, norm_interp_Y2])
    # print(interp_num)
    # print(norm_interp.shape)
    return norm_interp1, norm_interp2

def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length) 

def deleteRedundantCoor(x, y, maskTemp):
    mask = np.zeros_like(maskTemp)
    mask[x, y] = 1
    coor = np.nonzero(mask)
    return coor[0], coor[1]  # (2, n)

# print(generateDir([(0, 8), (13, 25), (31, 45)], [(8, 13), (25, 31), (45, 55)]))

###### Utils functions ######


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
        if not node_execute:
            self.minL = 1.5
        else:
            self.minL = 0.6
        self.maxL = 1.5
        self.minP = 0.5
        self.maxP = 0.8
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
        
        # update para
        self.local2global = None
        self.inner_idx = set(left_idx)
        self.outer_idx = set(right_idx)
    
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
        self.local2global_se3 = local2global
        self.global2local_se3 = np.linalg.inv(local2global)
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
        target_local = self.global2local_se3 @ np.array([target_global[0], target_global[1], 0, 1]) 
        cur_error = target_local[1]
        return cur_L, cur_P, cur_error, target_local, target_v, target_global, segment_end
    
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


class OGrid:
    def __init__(self, x_min=OGrid_x_min, x_max=OGrid_x_max, y_min=OGrid_y_min, y_max=OGrid_y_max, debug=False):
        self.r = OGrid_R
        # in local coor
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.w = int(np.ceil((self.y_max-self.y_min)/self.r + 1))  # x
        self.h = int(np.ceil((self.x_max-self.x_min)/self.r + 1))  # y
        print(f'grid_size: {self.w, self.h}')

        self.car_gridX = self.w // 2
        self.car_gridY = 0
        self.car_x_coor = self.car_gridX*self.r
        self.car_y_coor = 0
        self.center = np.array([self.car_x_coor, self.car_y_coor])
        self.nparrMap = np.zeros((self.w, self.h)).astype(np.int8)
        # self.map = MapMetaData()
        # self.map.width, self.map.height, self.map.resolution = w, h, self.r
        self.obstacles = []
        self.debug = debug

        # Lidar
        self.lidar_dmin = 0
        self.lidar_dmax = 20
        self.lidar_angular_resolution = 0.25
        self.lidar_angles = np.arange(-135,135, self.lidar_angular_resolution)*np.pi/180.0

    def local2grid(self, x, y):
        # coord: (2, n)
        new_x = -y + self.car_x_coor
        new_y = x 
        return new_x, new_y

    def grid2local(self, x, y):
        new_x = y
        new_y = -(x-self.car_x_coor)
        return new_x, new_y

    def gridCell_from_xy(self, x, y, duplicate=False):
        # x = np.clip(x, self.xmin, self.xmax)
        # y = np.clip(y, self.ymin, self.ymax)        
        if type(x) == np.float64 or type(x) == np.int32:
            x = np.array([x])
            y = np.array([y])
        # grid_cell = np.zeros((2, len(x))).astype(np.int32)
        grid_cell_x = np.floor(x/ self.r)
        grid_cell_y = np.floor(y/ self.r)
        if duplicate:
            x_center = np.floor(x/ self.r)
            y_center = np.floor(y/ self.r)
            grid_cell_x = np.hstack([x_center-1, x_center, x_center+1])
            grid_cell_y = np.hstack([y_center-1, y_center, y_center+1])
        # import ipdb; ipdb.set_trace()
        return grid_cell_x.astype(np.int32), grid_cell_y.astype(np.int32)

    # def get_OGridMsg(self):
    #     OGridMsg = OccupancyGrid()
    #     # import ipdb; ipdb.set_trace()
    #     OGridMsg.data = self.nparrMap.ravel().tolist()
    #     OGridMsg.info = self.map
    #     return OGridMsg

    def single_update_npMap(self, scan_d):
        scan_d = np.clip(scan_d, self.lidar_dmin, self.lidar_dmax)
        angles = self.lidar_angles
        # import ipdb; ipdb.set_trace()
        local_x = scan_d * np.cos(angles)  # 1-D array
        local_y = scan_d * np.sin(angles)  # 1-D array
        local_x = np.clip(local_x, self.x_min, self.x_max)
        local_y = np.clip(local_y, self.y_min, self.y_max)

        grid_x, grid_y = self.local2grid(local_x, local_y)
        grid_cell_x, grid_cell_y = self.gridCell_from_xy(grid_x, grid_y)
        self.nparrMap[grid_cell_x, grid_cell_y] = 1

    def update_npMap(self, scan_d):
        self.nparrMap = np.zeros((self.w, self.h)).astype(np.int8)
        self.single_update_npMap(scan_d)
        self.single_update_npMap(scan_d+np.ones_like(scan_d)*0.3)
        self.single_update_npMap(scan_d+np.ones_like(scan_d)*0.15)
        # self.nparrMap[grid_cell_x, np.clip(grid_cell_y+1, 0, self.h-1)] = 1
        # self.nparrMap[grid_cell_x, np.clip(grid_cell_y-1, 0, self.h-1)] = 1
        self.nparrMap[:, :2] = 0
        self.nparrMap[:, -2:] = 0

    def update_obstacle(self):
        self.obstacles = []
        obstacle = np.nonzero(self.nparrMap)
        x, y = obstacle[0], obstacle[1]
        local_x, local_y = self.grid2local(x*self.r, y*self.r)
        self.obstacles = np.vstack([local_x, local_y])
    
    def check_blocked(self, x, y, return_sum=False):

        local_x = np.clip(x, self.x_min, self.x_max)
        local_y = np.clip(y, self.y_min, self.y_max)

        grid_x, grid_y = self.local2grid(local_x, local_y)
        grid_cell_x, grid_cell_y = self.gridCell_from_xy(grid_x, grid_y)
        grid_cell_x, grid_cell_y = deleteRedundantCoor(grid_cell_x, grid_cell_y, self.nparrMap)
        # import ipdb;ipdb.set_trace()
        tmp_sum = np.sum(self.nparrMap[grid_cell_x, grid_cell_y])
        if  tmp_sum > 0:
            block = True
        else:
            block = False
        return block, tmp_sum
    
    # def find_line_max_gap(self, x, y):
    #     local_x = np.clip(x, self.x_min, self.x_max)
    #     local_y = np.clip(y, self.y_min, self.y_max)
    #     grid_x, grid_y = self.local2grid(local_x, local_y)
    #     grid_cell_x, grid_cell_y = self.gridCell_from_xy(grid_x, grid_y)
    #     block_mask = self.nparrMap([grid_cell_x, grid_cell_y] == 1)
    #     cur_gap, i, max_gap = 0, 0, 0
    #     n = len(block_mask)
    #     while i < n:
    #         if block_mask[i]:
    #             i += 1
    #         else:
    #             cur_gap = 0
    #             while i < n and not block_mask[i]:
    #                 cur_gap += 1
    #                 i += 1
    #             max_gap = max(max_gap, cur_gap)
    #     return max_gap

class RRTNode(object):
    def __init__(self, flow = None, parent = None, cost = None, is_root = False):
        self.flow = flow
        self.parent = parent
        self.cost = cost  # only used in RRT*
        self.is_root = is_root

# class def for RRT
class RRT(Node):
    def __init__(self):
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        super().__init__('RRT')
        self.scan_gap = 3
        self.L = 0.8
        self.P = 0.4
        self.nearst_idx = 0
        self.pathL = 7
        self.velocity = 2.0
        self.cur_position = np.zeros((2, 1))
        self.cur_yaw = 0

        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # TODO: create subscribers
        self.pose_sub_ = self.create_subscription(
            Odometry,
            pose_topic,
            self.pose_callback,
            1)
        
        self.scan_sub_ = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            1)
        
        # self.ogrid_pub = self.create_publisher(OccupancyGrid, ogrid_topic, 10)
        self.ogrid_markerpub = self.create_publisher(Marker, '/ogrid_marker', 10)
        # self.timer = self.create_timer(0.5, self.drawOgrid_callback)
        self.localgoal_markerpub = self.create_publisher(Marker, '/localg_marker', 10)
        self.rrtgoal_markerpub = self.create_publisher(Marker, '/rrtg_marker', 10)
        self.rrtpath_markerpub = self.create_publisher(Marker, '/rrtpath_marker', 10)
        self.waypoints_markerpub = self.create_publisher(Marker, '/wp_marker', 10)
        self.rrtplannedPath_markerpub = self.create_publisher(Marker, '/rrtplannedPath_marker', 10)

        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord) 
        # publishers
        # TODO: create a drive message publisher, and other publishers that you might need

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.Map_Processor = OGrid(debug=True)
        self.r = self.Map_Processor.r

        # ogridmarker
        self.ogridmarker = Marker(
                    type=Marker.POINTS,
                    id=0,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=Vector3(x=0.05, y=0.05, z=0.05),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),                    
                    )
        self.localgoalmarker = Marker(
                    type=Marker.SPHERE,
                    id=1,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=Vector3(x=0.2, y=0.2, z=0.2),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),                    
                    )
        self.rrtgoalmarker = Marker(
                    type=Marker.SPHERE,
                    id=2,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=Vector3(x=0.2, y=0.2, z=0.2),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),                    
                    )
        self.rrtpathmarker = Marker(
                    type=Marker.LINE_STRIP,
                    id=2,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=Vector3(x=0.01, y=0.1, z=0.1),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),                    
                    )
        self.rrtplannedPathmarker = Marker(
                    type=Marker.LINE_STRIP,
                    id=3,
                    # action = Marker.ADD, 
                    pose=Pose(),
                    scale=Vector3(x=0.01, y=0.1, z=0.1),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),                    
                    )

        # For pure pursuit
        self.planner = PurePursuitPlanner()
        
        # For RRT
        self.min_pos = -2.75
        self.max_pos = 2.75
        self.max_iters = 100
        self.max_dist = 0.1
        self.tree = []
        # self.goal_flow = self.sample()
        self.goal_threshold = 0.1
        self.path = []
        self.num_interp= 5

        self.prev_time = 0
        self.prev_scan_time = 0
        self.time_thres = 0.03

    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # calculate the distance from the wall

        cur_time = scan_msg.header.stamp.nanosec/1e9 + scan_msg.header.stamp.sec
        # print(f'scan_interval: {cur_time-self.prev_scan_time}')
        self.prev_scan_time = cur_time
        self.Map_Processor.update_npMap(scan_msg.ranges)

    
    def local2grid(self, coord):
        new_coord = np.zeros_like(coord)
        new_coord[0] = -coord[1] + self.Map_Processor.car_x_coor
        new_coord[1] = coord[0]
        return new_coord
    
    def grid2local(self, coord):
        new_coord = np.zeros_like(coord)
        new_coord[0] = coord[1]
        new_coord[1] = -(coord[0] - self.Map_Processor.car_x_coor)
        return new_coord

    def drive(self, steering_angle, velocity):
        self.ackermann_ord.drive.speed = velocity
        self.ackermann_ord.drive.steering_angle = steering_angle
        self.drive_publisher.publish(self.ackermann_ord)
    
    # def check_is_globalWp_blocked(self, segment_end):
    #     target_global_span = get_span_from_two_point(span_L=default_check_span, 
    #                                                  interp_num=check_interp_num,
    #                                                  pA=self.planner.wp[:2, safe_changeIdx(self.planner.wpNum, segment_end, -1)], 
    #                                                  pB=self.planner.wp[:2, segment_end])
    #     target_global_span = np.vstack([target_global_span, np.zeros((1, 3*check_interp_num)), np.ones((1, 3*check_interp_num))])
    #     target_local_span = self.planner.global2local_se3 @ target_global_span
    #     target_is_blocked, block_sum = self.Map_Processor.check_blocked(target_local_span[0], target_local_span[1])
    #     return target_is_blocked, block_sum
    
    def check_if_globalPoints_blocked(self, pA, pB, class_num=5, check_span=None):
        target_global_span = get_span_from_two_point(span_L=check_span, 
                                                     interp_num=check_interp_num,
                                                     pA=pA, 
                                                     pB=pB, class_num=class_num)
        # if return_points:                                             
        target_global_span = np.vstack([target_global_span, np.zeros((1, class_num*check_interp_num)), np.ones((1, class_num*check_interp_num))])
        target_local_span = self.planner.global2local_se3 @ target_global_span
        # import ipdb; ipdb.set_trace()
        target_is_blocked, block_sum = self.Map_Processor.check_blocked(target_local_span[0], target_local_span[1])
        return target_is_blocked, block_sum
    
    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:
        """
        cur_time = pose_msg.header.stamp.nanosec/1e9 + pose_msg.header.stamp.sec
        time_interval = cur_time - self.prev_time
        if time_interval < self.time_thres:
            return
        self.prev_time = cur_time
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
        self.cur_position = cur_position
        self.cur_yaw = yaw
        ####### get current pose and speed


        ##### global planning and update transformation matrix ###
        avoid_obs = False
        cur_L, cur_P, cur_error, target_local, target_v, target_global, segment_end = self.planner.find_targetWp(pose, speed)
        # cur_segment_end = segment_end
        
        # begin wp is two points eariler than end wp
        end_wp = self.planner.wp[:2, segment_end]
        begin_wp = self.planner.wp[:2, safe_changeIdx(self.planner.wpNum, segment_end, -2)]
        cur_blocked_1, _ = self.check_if_globalPoints_blocked(cur_position, target_global, class_num=9, check_span=default_check_span)
        cur_blocked_2, _ = self.check_if_globalPoints_blocked(begin_wp, target_global, class_num=9, check_span=default_check_span)
        
        # centerline is begin_wp to target_global
        centerline = np.linspace(begin_wp, target_global, num=check_interp_num)
        
        # import ipdb; ipdb.set_trace()
        if cur_blocked_1 or cur_blocked_2:
            # cur_segment_end = safe_changeIdx(self.planner.wpNum, cur_segment_end, 1)
            avoid_obs = True
        # print(f'avoid_obs: {avoid_obs}')
        # print(f'cur_position{cur_position}')
        # print(f'cur_block_sum{cur_block_sum}')

        ##### draw local goal Point
        self.localgoalmarker.pose.position.x = target_global[0]
        self.localgoalmarker.pose.position.y = target_global[1]
        self.localgoal_markerpub.publish(self.localgoalmarker)        
        ##### draw local goal Point

        if avoid_obs:
            # check two span 

            span1_pA, span1_pB, span2_pA, span2_pB = get_span_from_two_point(span_L=default_check_span, interp_num=check_interp_num, 
                                                                            pA=begin_wp, 
                                                                            pB=target_global,
                                                                            return_point=True)
            # normal_1, normal_2 = get_normal_from_two_point(pA=cur_position, pB=target_global)
            # normal1_local = self.planner.global2local_se3 @ normal_1
            # normal2_local = self.planner.global2local_se3 @ normal_2

            # # import ipdb; ipdb.set_trace()
            span1_blocked, span1_sum = self.check_if_globalPoints_blocked(span1_pA, span1_pB, class_num=9, check_span=default_check_span-check_offset)
            span2_blocked, span2_sum = self.check_if_globalPoints_blocked(span2_pA, span2_pB, class_num=9, check_span=default_check_span-check_offset)
            if span1_blocked and span2_blocked:
                cur_avoid_span = avoid_span
                choose = 2
            #     if safe_changeIdx(self.planner.wpNum, segment_end, -1) in self.planner.inner_idx:
            #         choose = 2
            #     else:
            #         choose = 1
            #     cur_avoid_span = avoid_span
            elif not span1_blocked:
                cur_avoid_span = avoid_span + avoid_offset
                choose=2
            elif not span2_blocked:
                cur_avoid_span = avoid_span - avoid_offset
                choose=2
            # if safe_changeIdx(self.planner.wpNum, segment_end, -1) in self.planner.inner_idx:
            #         choose = 2
            else:
                cur_avoid_span = avoid_span
                choose = 2
            
           
            # else:
            #     return
            #     cur_avoid_span = avoid_span + checking_offset
            #     if span1_sum > span2_sum:
            #     # next_wp = self.planner.wp[:2, safe_changeIdx(self.planner.wpNum, cur_segment_end, 1)]
            #     # if np.linalg.norm(next_wp-span1_pB) > np.linalg.norm(next_wp-span2_pB):
            #         choose = 2
            #     else:
            #         choose = 1
            print(f'avoid: {choose}, span is {cur_avoid_span}, segment_end is {segment_end}')
            print(f'blocked:{avoid_obs}, span1 is blocked: {span1_blocked}, span2 is blocked: {span2_blocked}')
            
            # get new span
            span1_pA, span1_pB, span2_pA, span2_pB = get_span_from_two_point(span_L=default_check_span, interp_num=2, 
                                                                            pA=centerline[-2], 
                                                                            pB=centerline[-1],
                                                                            return_point=True)
            
            if choose == 2:
                _, _, target_pA, target_pB = get_span_from_two_point(span_L=cur_avoid_span, interp_num=2, 
                                                                    pA=span2_pA, 
                                                                    pB=span2_pB,
                                                                    return_point=True)
            else:
                target_pA, target_pB, _, _ = get_span_from_two_point(span_L=cur_avoid_span, interp_num=2, 
                                                                    pA=span1_pA, 
                                                                    pB=span1_pB,
                                                                    return_point=True)
            target_global = (target_pA + target_pB) / 2 
            target_global = np.array([target_global[0], target_global[1], 0, 1])
            cur_L = np.linalg.norm(target_global[:2]-cur_position)
            cur_P = self.planner.maxP
            target_local = self.planner.global2local_se3 @ target_global
            cur_error = target_local[1]
            steering_angle, velocity = self.planner.planning(cur_L, cur_P, cur_error, target_local, target_v, time_interval)
        else:
            steering_angle, velocity = self.planner.planning(cur_L, cur_P, cur_error, target_local, target_v, time_interval)
        
        if node_execute:
            self.drive(steering_angle, velocity)
        else:
            if avoid_obs:
                # print(f'target_global:{target_global}')
                # print(f'span1: {span1_blocked, span1_pA, span1_pB}')
                # print(f'span2: {span2_blocked, span2_pA, span2_pB}')
                # print(f'avoid_obs: {avoid_obs}')
                # print(f'cur_segment: {cur_segment_end}, original_segment: {segment_end}')
                pass
        self.prev_time = cur_time
        ##### global planning and update transformation matrix ###
        
        # RRT, get rrt goal
        # rrt_goal = np.array([1.0, 1.0])
        # self.Map_Processor.update_obstacle()
        # if self.plan(self.Map_Processor.center, self.local2grid(target_local[:2]), self.Map_Processor.nparrMap):
        #     # print('find')
        #     # ipdb.set_trace()
        #     if len(self.path) != 0:
        #         rrt_goal = self.path[min(self.pathL, len(self.path)-1)]
        #         rrt_goal = self.grid2local(rrt_goal)
        # # # # pure pursuit
        # # cur_L = np.linalg.norm(rrt_goal)
        
        # if node_execute:
        #     self.pure_pursuit(rrt_goal, cur_L)

        ##### draw Ogrid
        self.Map_Processor.update_obstacle()
        self.ogridmarker.points = []
        for i, point in enumerate(self.Map_Processor.obstacles.T):
            x, y = point[0], point[1]
            local_obs = np.array([x, y, 0, 1])
            global_obs = self.planner.local2global_se3 @ local_obs
            # print(x, y)
            point = Point()
            point.x, point.y, point.z = global_obs[0], global_obs[1], 0.0
            self.ogridmarker.points.append(point)
        self.ogrid_markerpub.publish(self.ogridmarker)
        ##### draw Ogrid



        ##### draw goal Point
        x, y = target_global[0], target_global[1]
        # local_obs = np.array([x, y, 0, 1])
        # global_obs = self.planner.local2global_se3 @ local_obs        
        # self.rrtgoalmarker.pose.position.x = global_obs[0]
        # self.rrtgoalmarker.pose.position.y = global_obs[1]
        self.rrtgoalmarker.pose.position.x = target_global[0]
        self.rrtgoalmarker.pose.position.y = target_global[1]
        # self.localgoal_markerpub.publish(self.rrtgoalmarker)
        self.rrtgoal_markerpub.publish(self.rrtgoalmarker)
        # ##### draw rrt goal Point         
        
    #     # #### draw rrt path
    #     # self.rrtplannedPathmarker.points = []
    #     # for i, flow in enumerate(self.path):
    #     #     flow = self.grid2local(flow)
    #     #     x, y = flow[0], flow[1]
    #     #     local_obs = np.array([x, y, 0, 1])
    #     #     global_obs = local2global @ local_obs
    #     #     # print(x, y)
    #     #     point = Point()
    #     #     point.x = global_obs[0]
    #     #     point.y = global_obs[1]
    #     #     point.z = 0.0
    #     #     self.rrtplannedPathmarker.points.append(point)
    #     # self.rrtplannedPath_markerpub.publish(self.rrtplannedPathmarker)

    #     # self.rrtpathmarker.points = []
    #     # for i, flow in enumerate(self.path):
    #     #     flow = self.grid2local(flow)
    #     #     x, y = flow[0], flow[1]
    #     #     local_obs = np.array([x, y, 0, 1])
    #     #     global_obs = local2global @ local_obs
    #     #     # print(x, y)
    #     #     point = Point()
    #     #     point.x = global_obs[0]
    #     #     point.y = global_obs[1]
    #     #     point.z = 0.0
    #     #     self.rrtpathmarker.points.append(point)
    #     # self.rrtpathmarker.points = []
    #     # for pose in self.tree:
    #     #     x1, y1 = self.grid2local(pose[0])[0], self.grid2local(pose[0])[1]
    #     #     # x2, y2 = self.grid2local(pose[1])[0], self.grid2local(pose[1])[1]
    #     #     local_obs = np.array([x1, y1, 0, 1])
    #     #     global_obs = local2global @ local_obs
    #     #     # print(x, y)
    #     #     point = Point()
    #     #     point.x = global_obs[0]
    #     #     point.y = global_obs[1]
    #     #     point.z = 0.0
    #     #     self.rrtpathmarker.points.append(point)            
    #     # self.rrtpath_markerpub.publish(self.rrtpathmarker)                  
    
    # def sample(self):
    #     """
    #     This method should randomly sample the free space, and returns a viable point

    #     Args:
    #     Returns:
    #         (x, y) (float float): a tuple representing the sampled point

    #     """
    #     coord = np.zeros((2))
    #     coord[0] = np.random.uniform(0, self.max_pos)
    #     coord[1] = np.random.uniform(0, self.max_pos)
    #     # y = None
    #     return coord
    
    # def interp_between_2points(self,sample_point,parent_point):
    #     # print("!",sample_point[0],parent_point.shape)
    #     x_interp=np.linspace(sample_point[0],parent_point[0],self.num_interp)
    #     y_interp = np.linspace(sample_point[1], parent_point[1], self.num_interp)
    #     interp_point=np.vstack((x_interp, y_interp)).T
    #     # print("points",interp_point.shape)
    #     return interp_point

    # def check_free_space(self,interp_points_idx,grid_map): # True means free space, False means obstacle
    #     idx=interp_points_idx.T  # (2, n)
    #     # import ipdb; ipdb.set_trace()
    #     num=np.count_nonzero(grid_map[idx[0], idx[1]])
    #     if num==0:
    #         return True
    #     else:
    #         return False

    # def convert_coord2Index(self, coord, grid_map):
    #     index = np.floor(coord / self.r).astype(np.int64)
    #     return index
    
    # def get_path(self, node):
    #     path = [node.flow]
    #     while node.parent is not None:
    #         node = node.parent
    #         path.append(node.flow)
    #     path.reverse()

    #     return path

    # def get_nearest_node_by_dist(self, flows, target, dict):
    #     diff = flows - target
    #     dist = np.linalg.norm(diff, axis = 1)
    #     idx = np.argmin(dist)
    #     return dict[idx]
    
    # def scale_target_flows(self, target_flows, parent_node):
    #     # print("----",target_flows,parent_node)
    #     dist = np.linalg.norm(target_flows - parent_node.flow)
    #     # print("1",dist)
    #     if dist > self.max_dist:
    #         target_flows = parent_node.flow + (target_flows - parent_node.flow) / dist * self.max_dist
    #     return target_flows

    # def plan(self, start, goal, grid_map):
    #     """
    #     RRT plan in grid world axis
    #     """
    #     # print(start)
    #     # print(goal)
    #     # ipdb.set_trace()
    #     first_flow = start
    #     self.goal_flow = goal
    #     # grid_map = self.init_gridMap()
    #     first_node = RRTNode(flow = first_flow, parent = None, cost = None, is_root = True)
    #     self.tree = []
    #     self.nodes = [first_node]

    #     dic = {}
    #     node = first_node
    #     temp = node.flow.reshape(1, 2)
    #     dic[temp.shape[0] - 1] = first_node

    #     for i in range(5000):
    #         if np.random.random() > 0.1:
    #             target_flow = self.sample()
    #         else:
    #             target_flow = goal
    #         parent_node = self.get_nearest_node_by_dist(temp, target_flow, dic)
    #         target_flow = self.scale_target_flows(target_flow, parent_node)

    #         interp_points=self.interp_between_2points(target_flow,parent_node.flow)
    #         interp_points_index=self.convert_coord2Index(interp_points,grid_map)
    #         # import ipdb;
    #         # ipdb.set_trace()
    #         if self.check_free_space(interp_points_index,grid_map):

    #             new_node = RRTNode(flow = target_flow, parent = parent_node, cost = None, is_root = False)
    #             self.tree.append(((new_node.flow[0], new_node.flow[1]),
    #                                 (parent_node.flow[0], parent_node.flow[1])))
    #             node = new_node

    #             # print(temp.shape,temp)
    #             # print(node.flow.shape,node.flow)
    #             t = node.flow.reshape(1, 2)
    #             temp = np.concatenate((temp, t), axis = 0)
    #             # print("temp",temp)
    #             dic[temp.shape[0] - 1] = node
    #             # print(temp.shape,temp)
    #             # temp = torch.cat((temp, torch.unsqueeze(node.flows, dim = 0)), 0)

    #             diff = target_flow - self.goal_flow
    #             dist_ = np.linalg.norm(diff)
    #             # print(dist_)
    #             if dist_ < self.goal_threshold:
    #                 # print("find")
    #                 self.path = self.get_path(new_node)
    #                 # print(path)
    #                 return True
    #     return False


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()