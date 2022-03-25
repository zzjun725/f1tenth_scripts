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

# TODO: import as you need

# class def for tree nodes
# It's up to you if you want to use this
home = '/sim_ws'
os.makedirs(home+'/wp_log', exist_ok = True)
log_position = home+'/wp_log'
for file in os.listdir(log_position):
    if file.startswith('wp'):
        wp_log = csv.reader(open(os.path.join(log_position, file)))
        # import ipdb; ipdb.set_trace()
        print('load wp_log')
waypoints = []
for i, row in enumerate(wp_log):
    # print(row)
    if (i % 50 != 0):
        continue
    if len(row) > 2:
        x, y = row[0], row[1]
        x, y = float(x), float(y)
        waypoints.append(np.array([x, y]))
wp = np.array(waypoints)
print(len(wp))

class OGrid:
    def __init__(self, h=60, w=41, debug=False):
        self.h = h
        self.w = w
        self.r = 3.0/w
        self.x_center = self.w // 2
        self.y_center = 0
        self.x_coord = self.x_center*self.r
        self.y_coord = 0
        self.goal_threshold = 0.1
        self.center = np.array([self.x_coord, self.y_coord])
        self.nparrMap = np.zeros((w, h)).astype(np.int8)
        # self.map = MapMetaData()
        # self.map.width, self.map.height, self.map.resolution = w, h, self.r
        self.obstacles = []
        self.debug = debug

    # def get_OGridMsg(self):
    #     OGridMsg = OccupancyGrid()
    #     # import ipdb; ipdb.set_trace()
    #     OGridMsg.data = self.nparrMap.ravel().tolist()
    #     OGridMsg.info = self.map
    #     return OGridMsg
    def clip_index(self, data, down, upper):
        return np.clip(data, down, upper)

    def update_npMap(self, theta_dist: dict):
        """
        Input: theta_dist: {theta: dist}
        """
        self.obstacles = []
        x_center = self.x_center
        y_center = self.y_center
        self.nparrMap = np.zeros_like(self.nparrMap)
        for th, d in theta_dist.items():
            y_incre = int(cos(th) * d / self.r)
            x_incre = int(-sin(th) * d / self.r)
            if 0 <= y_incre < self.h and -self.w//2+1 <= x_incre < self.w//2:
                target_x, target_y = x_center + x_incre, y_center + y_incre
                self.nparrMap[target_x][target_y] = 1
                for i, j in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                    self.nparrMap[self.clip_index(target_x + i, 0, self.w-1)][self.clip_index(target_y + j, 0, self.h-1)] = 1

        # self.nparrMap.astype(np.int8)
        # if self.debug:
        #     print(self.nparrMap.T)

    def update_obstacle(self):
        self.obstacles = []
        obstacle = np.nonzero(self.nparrMap)
        for i in range(len(obstacle[0])):
            x, y = obstacle[0][i], obstacle[1][i]
            local_x = (x-self.x_center) * self.r
            local_y = (y-self.y_center) * self.r
            local_x, local_y = local_y, -local_x
            self.obstacles.append([local_x, local_y])

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
        pose_topic = "pf/pose/odom"
        scan_topic = "/scan"
        # ogrid_topic = '/ogrid'
        drive_topic = '/drive'
        self.scan_gap = 3
        self.L = 1.0
        self.P = 0.4
        self.nearst_idx = 0
        self.pathL = 2
        self.velocity = 2.0
        self.cur_position = np.zeros((2, 1))
        self.cur_yaw = 0
        self.circle_radius = 0.5

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

        # For RRT
        self.min_pos = -2.75
        self.max_pos = 2.75
        self.max_iters = 100
        self.max_dist = 0.1
        self.tree = []
        self.goal_flow = self.sample()
        self.goal_threshold = 0.1
        self.path = []
        self.num_interp= 5

                

    def scan_processor(self, scan_msg, gap=5):
        """
        return: res{theta: dist}
        """
        res = dict()
        angle_increment = scan_msg.angle_increment
        angle_min = scan_msg.angle_min
        ranges = scan_msg.ranges
        for i in range(170, 905, 5):
            cur_angle = i*angle_increment + angle_min
            res[cur_angle] = ranges[i]
        return res    

    
    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        # calculate the distance from the wall
        theta_dist = self.scan_processor(scan_msg, gap=self.scan_gap)
        self.Map_Processor.update_npMap(theta_dist)
    
    def local2grid(self, coord):
        new_coord = np.zeros_like(coord)
        new_coord[0] = -coord[1] + self.Map_Processor.x_coord
        new_coord[1] = coord[0]
        return new_coord
    
    def grid2local(self, coord):
        new_coord = np.zeros_like(coord)
        new_coord[0] = coord[1]
        new_coord[1] = -(coord[0] - self.Map_Processor.x_coord)
        return new_coord

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:
        """
        
        # get localgoal from waypoints
        cur_position = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        near_dist = 100   
        for i, point in enumerate(wp):  # (x, y)
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist < near_dist:
                near_dist = cur_dist
                self.nearst_idx = i
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
        local_goalP = np.linalg.inv(local2global) @ np.array([interp_point[0], interp_point[1], 0, 1])

        rrt_goal = local_goalP[:2]

        self.cur_position = cur_position
        self.cur_yaw = yaw
        
        # RRT, get rrt goal 
        self.Map_Processor.update_obstacle()
        if self.plan(self.Map_Processor.center, self.local2grid(local_goalP[:2]), self.Map_Processor.nparrMap):
            # print('find')
            # ipdb.set_trace()
            if len(self.path) != 0:
                rrt_goal = self.path[min(self.pathL, len(self.path)-1)]
                rrt_goal = self.grid2local(rrt_goal)
        # # # pure pursuit
        cur_L = np.linalg.norm(rrt_goal)
        # cur_L = np.linalg.norm(local_goalP[:2])
        # print(local_goalP[:2])
        # print(rrt_goal)
        # ipdb.set_trace()
        # self.pure_pursuit(rrt_goal, cur_L)

        # draw Ogrid
        self.ogridmarker.points = []
        for i, point in enumerate(self.Map_Processor.obstacles):
            x, y = point[0], point[1]
            local_obs = np.array([x, y, 0, 1])
            global_obs = local2global @ local_obs
            # print(x, y)
            point = Point()
            point.x = global_obs[0]
            point.y = global_obs[1]
            point.z = 0.0
            self.ogridmarker.points.append(point)
        self.ogrid_markerpub.publish(self.ogridmarker)
        
        # draw local goal Point
        self.localgoalmarker.pose.position.x = interp_point[0]
        self.localgoalmarker.pose.position.y = interp_point[1]
        self.localgoal_markerpub.publish(self.localgoalmarker)        
        
        # draw rrt goal Point
        x, y = rrt_goal[0], rrt_goal[1]
        local_obs = np.array([x, y, 0, 1])
        global_obs = local2global @ local_obs        
        self.rrtgoalmarker.pose.position.x = global_obs[0]
        self.rrtgoalmarker.pose.position.y = global_obs[1]
        # self.localgoal_markerpub.publish(self.rrtgoalmarker)
        self.rrtgoal_markerpub.publish(self.rrtgoalmarker)         
        
        self.rrtpathmarker.points = []
        for i, flow in enumerate(self.path):
            flow = self.grid2local(flow)
            x, y = flow[0], flow[1]
            local_obs = np.array([x, y, 0, 1])
            global_obs = local2global @ local_obs
            # print(x, y)
            point = Point()
            point.x = global_obs[0]
            point.y = global_obs[1]
            point.z = 0.0
            self.rrtpathmarker.points.append(point)
        self.rrtpathmarker.points = []
        for pose in self.tree:
            x1, y1 = self.grid2local(pose[0])[0], self.grid2local(pose[0])[1]
            # x2, y2 = self.grid2local(pose[1])[0], self.grid2local(pose[1])[1]
            local_obs = np.array([x1, y1, 0, 1])
            global_obs = local2global @ local_obs
            # print(x, y)
            point = Point()
            point.x = global_obs[0]
            point.y = global_obs[1]
            point.z = 0.0
            self.rrtpathmarker.points.append(point)            
        self.rrtpath_markerpub.publish(self.rrtpathmarker)             
        
        # OGridMsg = self.Map_Processor.get_OGridMsg()
        # OGridMsg.info.origin = Pose()
        # OGridMsg.info.origin.position.x = pose_msg.pose.pose.position.x
        # OGridMsg.info.origin.position.y = pose_msg.pose.pose.position.y
        # OGridMsg.info.origin.orientation.w = pose_msg.pose.pose.orientation.w
        # OGridMsg.info.origin.orientation.x = pose_msg.pose.pose.orientation.x
        # OGridMsg.info.origin.orientation.y = pose_msg.pose.pose.orientation.y
        # OGridMsg.info.origin.orientation.z = pose_msg.pose.pose.orientation.z
        # self.ogrid_pub.publish(OGridMsg)

    def drawOgrid_callback(self):
        # Visualization
        yaw, cur_position = self.cur_yaw, self.cur_position
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, cur_position[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, cur_position[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        self.ogridmarker.points = []
        for i, point in enumerate(self.Map_Processor.obstacles):
            x, y = point[0], point[1]
            local_obs = np.array([x, y, 0, 1])
            global_obs = local2global @ local_obs
            # print(x, y)
            point = Point()
            point.x = global_obs[0]
            point.y = global_obs[1]
            point.z = 0.0
            self.ogridmarker.points.append(point)
        self.ogrid_markerpub.publish(self.ogridmarker)
        print('publish ogrid')  

    
    def pure_pursuit(self, local_goalP, cur_L):
        gamma = 2*abs(local_goalP[1]) / (cur_L ** 2)
        if local_goalP[1] > 0:
            steering_angle = self.P * gamma
        else:
            steering_angle = self.P * -gamma
        velocity = self.velocity
        if abs(steering_angle) >=1:
            steering_angle = steering_angle / 2
        
        self.ackermann_ord.drive.speed = velocity
        self.ackermann_ord.drive.steering_angle = steering_angle
        self.drive_publisher.publish(self.ackermann_ord)        
    
    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        coord = np.zeros((2))
        coord[0] = np.random.uniform(0, self.max_pos)
        coord[1] = np.random.uniform(0, self.max_pos)
        # y = None
        return coord
    
    def interp_between_2points(self,sample_point,parent_point):
        # print("!",sample_point[0],parent_point.shape)
        x_interp=np.linspace(sample_point[0],parent_point[0],self.num_interp)
        y_interp = np.linspace(sample_point[1], parent_point[1], self.num_interp)
        interp_point=np.vstack((x_interp, y_interp)).T
        # print("points",interp_point.shape)
        return interp_point

    def check_free_space(self,interp_points_idx,grid_map): # True means free space, False means obstacle
        idx=interp_points_idx.T
        num=np.count_nonzero(grid_map[list(idx)])
        if num==0:
            return True
        else:
            return False

    def convert_coord2Index(self, coord, grid_map):
        index = np.floor(coord / self.r).astype(np.int64)
        return index
    
    def get_path(self, node):
        path = [node.flow]
        while node.parent is not None:
            node = node.parent
            path.append(node.flow)
        path.reverse()

        return path

    def get_nearest_node_by_dist(self, flows, target, dict):
        diff = flows - target
        dist = np.linalg.norm(diff, axis = 1)
        idx = np.argmin(dist)
        return dict[idx]
    
    def scale_target_flows(self, target_flows, parent_node):
        # print("----",target_flows,parent_node)
        dist = np.linalg.norm(target_flows - parent_node.flow)
        # print("1",dist)
        if dist > self.max_dist:
            target_flows = parent_node.flow + (target_flows - parent_node.flow) / dist * self.max_dist
        dist = np.linalg.norm(target_flows - parent_node.flow)
        return target_flows, dist

    def rewire(self,idx_in_dict, cost_, dist_,dict,new_node,grid_map):
        if idx_in_dict is not None:
            current_node_cost=new_node.cost
            other_nodes_cost = np.array(cost_)
            dist_btw_circle_node=dist_[idx_in_dict] # all distance bewteen currect node and all other node in circle
            circle_node_cost=other_nodes_cost[idx_in_dict]
            #print("currnet_nodes_cost",current_node_cost)
            potential_cost=dist_btw_circle_node+current_node_cost

            #print(potential_cost.shape,circle_node_cost.shape)
            diff=potential_cost-circle_node_cost
            #print("-------------",np.where(diff<0.)[0])
            idx=np.where(diff<0.)[0]
            for id in idx_in_dict[idx]:
                interp_points = self.interp_between_2points(dict[id].flow, new_node.flow)
                interp_points_index = self.convert_coord2Index(interp_points, grid_map)
                if self.check_free_space(interp_points_index, grid_map):
                    dict[id].cost=dist_[id]+current_node_cost
                    dict[id].parent=new_node
            # for node in dict[idx_in_dict[idx]]:
            #     node.cost=potential_cost[idx]
            #     node.parent=new_node

    def get_parent_by_pathCost(self, flows,target, dict, cost):
        cost_ = np.array(cost)
        # print("cost",cost_)
        diff = flows - target
        dist = np.linalg.norm(diff, axis = 1)
        # print("diff",dist)
        # print(np.where(dist<0.)[0])

        index_in_dict = np.where(dist < self.circle_radius)[0]

        dist_from_target_root = dist[index_in_dict] + cost_[index_in_dict]
        # print("dist_from_target_root",dist_from_target_root)
        if (len(list(dist_from_target_root)) != 0):
            idx = np.argmin(dist_from_target_root)  # the idx is the index of index_in_dict, not in dict
            # print()
            return dict[index_in_dict[idx]], dist[index_in_dict][idx], index_in_dict,dist
            # print("idx",idx)
        else:
            idx = np.argmin(dist)
            return dict[idx], dist[idx], None,dist

    def plan(self, start, goal, grid_map):
        """
        RRT plan in grid world axis
        """
        # print(start)
        # print(goal)
        # ipdb.set_trace()
        first_flow = start
        self.goal_flow = goal
        # grid_map = self.init_gridMap()
        first_node = RRTNode(flow = first_flow, parent = None, cost = 0, is_root = True)
        self.tree = []
        self.nodes = [first_node]
        
        dic = {}
        node = first_node
        temp = node.flow.reshape(1, 2)
        dic[temp.shape[0] - 1] = first_node
        cost_ = [node.cost]

        for i in range(5000):
            if np.random.random() > 0.5:
                target_flow = goal
            else:
                target_flow = self.sample()
            parent_node = self.get_nearest_node_by_dist(temp, target_flow, dic)
            target_flow, dist_btw = self.scale_target_flows(target_flow, parent_node)

            parent_node, dist_btw, idx_of_dic_in_nodes_Cirlce,dist_all = self.get_parent_by_pathCost(temp, target_flow, dic,cost_)
            # ipdb.set_trace()
            interp_points = self.interp_between_2points(target_flow, parent_node.flow)
            interp_points_index = self.convert_coord2Index(interp_points, grid_map)
            if self.check_free_space(interp_points_index, grid_map):
                # print("----------------------------------")
                new_node = RRTNode(flow = target_flow, parent = parent_node, cost = parent_node.cost + dist_btw,
                                is_root = False)
                ########### rewire ##############

                self.rewire(idx_of_dic_in_nodes_Cirlce, cost_, dist_all,dic,new_node,grid_map)

                #################################
                self.tree.append(((new_node.flow[0], new_node.flow[1]),
                                  (parent_node.flow[0], parent_node.flow[1])))
                node = new_node

                # print(temp.shape,temp)
                # print(node.flow.shape,node.flow)
                t = node.flow.reshape(1, 2)
                temp = np.concatenate((temp, t), axis = 0)
                # print("temp",temp)
                dic[temp.shape[0] - 1] = node
                cost_.append(node.cost)
                # print(temp.shape,temp)
                # temp = torch.cat((temp, torch.unsqueeze(node.flows, dim = 0)), 0)

                diff = target_flow - self.goal_flow
                dist_ = np.linalg.norm(diff)
                # print(dist_)
                if dist_ < self.goal_threshold:
                    # print("find")
                    self.path = self.get_path(new_node)
                    # print(path)
                    return True
        return False


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
