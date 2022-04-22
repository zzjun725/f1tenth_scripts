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
os.makedirs(home+'/wp_log', exist_ok = True)
log_position = home+'/wp_log'
wp_path = log_position
wp_gap = 1
node_publish = False
node_execute = True
# load_gap = 20
# for file in os.listdir(log_position):
#     if file.startswith('wp'):
#         wp_log = csv.reader(open(os.path.join(log_position, file)))
#         # import ipdb; ipdb.set_trace()
#         print('load wp_log')
# waypoints = []
# last_x, last_y = 0, 0
# for i, row in enumerate(wp_log):
#     if i % load_gap ==0:
#         x, y, v = row[0], row[1], row[2]
#         x, y, v = float(x), float(y), float(v)
#         waypoints.append(np.array([x, y, v]))
# wp = np.array(waypoints[:-1])

# arcPointsNum = 15
# LongPointsNum = 40
# ShortPointsNum = 20
# print(len(wp), wp.shape)


class TrackingPlanner:
    def __init__(self, wp_path=wp_path, wp_gap = 1, debug=True):
        # wp
        self.wp = []
        self.wpNum = None
        self.wp_path = wp_path
        self.wpGapCounter = 0
        self.wpGapThres = wp_gap
        self.max_speed = 70
        self.speedScale = 1.0


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
        # TODO: fix this
        for i, row in enumerate(wp_log):
            if (i % wp_gap) == 0:
                points.append([float(row[0]), float(row[1]), float(row[2])])
        self.wp = np.array(points).T
        print(f'shape of wp: {self.wp.shape}')

    
    def load_Optimalwp(self): 
        for file in os.listdir(wp_path):
            if file.startswith('Optimalwp'):
                wp_log = csv.reader(open(os.path.join(wp_path, file)))
                print('load Optimalwp_log')
                break
        for i, row in enumerate(wp_log):
            if i > 2:
                # import ipdb; ipdb.set_trace()
                if self.wpGapCounter == self.wpGapThres:
                    self.wpGapCounter = 0
                    row = row[0].split(';')
                    x, y, v = float(row[1]), float(row[2]), np.clip(float(row[5])/self.speedScale, 0, self.max_speed)
                    self.wp.append([x, y, v])
                    # TODO: fix this if the road logger is correct
                    # self.wp.append([y, x, v])
                else:
                    self.wpGapCounter += 1
        self.wp = np.array(self.wp).T  # (3, n), n is the number of waypoints
        self.wpNum = len(self.wp[0])
    
    def pidAccel(self, diff, targetS=0, curS=0):
        a = self.P * diff
        if a > 0 :
            a = self.Increase_P * a
        else: 
            a = self.Decrease_P * a
        print(f'a: {np.round(a, 3)}')
        a = np.clip(a, -1.0, 1.0)
        print(f'a: {np.round(a, 3)}')
        return np.clip(a, -1.0, 1.0)
    
    
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
        self.minL = 0.4
        self.maxL = 2.0
        self.minP = 0.4
        self.maxP = 0.9
        self.interpScale = 20
        self.Pscale = 8
        self.Lscale = 8
        self.interp_P_scale = (self.maxP-self.minP) / self.Pscale
        self.interp_L_scale = (self.maxL-self.minL) / self.Lscale
        self.prev_error = 0
        self.D = 0
        self.errthres = 0.1
        # self.load_Optimalwp()  # (3, n), n is the number of waypoints
        self.load_wp()
    
    def find_targetWp(self, cur_position, targetL):
        """
        cur_positon: (2, )
        return: cur_L, targetWp(2, ), targetV 
        """
        # ipdb.set_trace()

        wp_xyaxis = self.wp[:2]  # (2, n)
        dist = np.linalg.norm(wp_xyaxis-cur_position.reshape(2, 1), axis=0)
        nearst_idx = np.argmin(dist)
        nearst_point = wp_xyaxis[:, nearst_idx]

        # print(nearst_idx)
        segment_end = nearst_idx
        find_p = False
        for i, point in enumerate(wp_xyaxis.T[nearst_idx:]):
            cur_dist = np.linalg.norm(cur_position-point)
            if cur_dist > targetL:
                find_p = True
                break
        if not find_p:
            # import ipdb; ipdb.set_trace()
            for i, point in enumerate(wp_xyaxis.T[:]):
                cur_dist = np.linalg.norm(cur_position-point)
                if cur_dist > targetL:
                    segment_end = i
                    break           
        else:
            segment_end += i
        interp_point = np.array([wp_xyaxis[0][segment_end], wp_xyaxis[1][segment_end]])
        interp_v = self.wp[2][segment_end]
        # get interpolation
        # error = 0.1
        if segment_end != 0:
            x_array = np.linspace(wp_xyaxis[0][segment_end-1], wp_xyaxis[0][segment_end], self.interpScale)
            y_array = np.linspace(wp_xyaxis[1][segment_end-1], wp_xyaxis[1][segment_end], self.interpScale)
            v_array = np.linspace(self.wp[2][segment_end-1], self.wp[2][segment_end], self.interpScale)
            xy_interp = np.vstack([x_array, y_array])
            dist_interp = np.linalg.norm(xy_interp-cur_position.reshape(2, 1), axis=0) - targetL
            i_interp = np.argmin(np.abs(dist_interp))
            interp_point = np.array([x_array[i_interp], y_array[i_interp]])
            interp_v = v_array[i_interp]
        cur_L = np.linalg.norm(cur_position-interp_point)
        # ipdb.set_trace()
        # print(segment_end)
        return cur_L, interp_point, interp_v, segment_end, nearst_point
    
    def planning(self, pose, speed, time_interval=None):
        """
        pose: (global_x, global_y, yaw) of the car
        """
        targetL = speed * self.interp_L_scale + self.minL
        P = self.maxP - speed * self.interp_P_scale 
        
        yaw = pose[2]
        local2global = np.array([[np.cos(yaw), -np.sin(yaw), 0, pose[0]], 
                                 [np.sin(yaw), np.cos(yaw), 0, pose[1]], 
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])        
        
        # wp_xyaxis = self.wp[:2]
        cur_L, targetWp, targetV, segment_end, nearstP = self.find_targetWp(pose[:2], targetL)

        global2local = np.linalg.inv(local2global)
        nearstP_local = global2local @ np.array([nearstP[0], nearstP[1], 0, 1]) 
        cur_error = nearstP_local[1]

        if time_interval:
            offset = self.D * (cur_error - self.prev_error) / time_interval
        else:
            offset = self.D * (cur_error - self.prev_error)
        self.prev_error = cur_error
        # print(f'D_offset: {offset}')

        local_goalP = global2local @ np.array([targetWp[0], targetWp[1], 0, 1])
        gamma = 2*abs(local_goalP[1]) / (cur_L ** 2)
        if local_goalP[1] < 0:
            steering_angle = P * -gamma
        else:
            steering_angle = P * gamma
        steering_angle = np.clip(steering_angle+offset, -1.0, 1.0)
        # self.targetSpeed = targetV
        # diff = targetV - speed
        # acceleration = self.pidAccel(diff)

        return steering_angle, targetV, targetWp    


class StanleyPlanner(TrackingPlanner):
    def __init__(self, wp_path=wp_path, wp_gap=3, 
                 drawW=400, drawH=800, drawLen=10, debug=True,
                 wb=0.3302, kv=8):
        super().__init__(wp_path, wp_gap, drawW, drawH, drawLen, debug)
        self.wheelbase = wb
        self.kv = kv
        self.load_Optimalwp()
        self.P = 0.4
        self.headSpan = 2

    def _get_average_k(self, p):
        p1 = p[:, :-1]  # (2, n-1)
        p2 = p[:, 1:]  # (2, n-1)
        delta = p2 - p1 # (2, n-1)
        k = np.arctan2(delta[1], delta[0])
        k = (k + 2*np.pi) % (2*np.pi)
        k = np.mean(k)
        return k
    
    def _get_current_waypoint_and_heading(self, position):
        wpts = self.wp[:2]
        # nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts.T)
        wp_xyaxis = self.wp[:2]  # (2, n)
        dist = np.linalg.norm(wp_xyaxis-position.reshape(2, 1), axis=0)
        i = np.argmin(dist)
        nearest_point = wp_xyaxis[:, i]  # (2, )
       
        begin = np.clip(i-self.headSpan, 0, self.wpNum-2)
        end = np.clip(i+self.headSpan, 1, self.wpNum-1)
        heading = self._get_average_k(wp_xyaxis[:, begin:end+1])

        return nearest_point, heading, i 

    @staticmethod
    def _calculate_angle_difference(start, end):
        a = end - start
        # a = (a + np.pi) % (2*np.pi) - np.pi
        return a

    def planning(self, pose, speed):
        position = np.array([pose[0], pose[1]])  # (2, )
        cur_heading = (-pose[2] + np.pi/2) % (2*np.pi)  # (y is the heading)
        nearest_point, target_heading, i = self._get_current_waypoint_and_heading(position)
        # target_heading = (target_heading + 2*np.pi) % (2*np.pi)
        
        # speed 
        target_speed = self.wp[2][i]
        accelation = self._acceleration_logic(target_speed, speed, self.max_speed)

        # the cur_heading and target_heading all in (0, 2*np.pi)
        heading_error = self._calculate_angle_difference(cur_heading, target_heading)
        
        # distance error
        d_nearest = nearest_point - position
        y_Unitvector = np.array([np.cos(cur_heading), np.sin(cur_heading)])
        nearest_dist = np.dot(d_nearest, y_Unitvector)

        # nearest_dist = np.dot(d_nearest, [np.cos(pose[2] + np.pi / 2), np.sin(pose[2] + np.pi / 2)])
        # ipdb.set_trace()
        distance_error = np.arctan2(self.kv*nearest_dist, speed+1)
        steering_angle = heading_error+distance_error/2
        steering_angle = steering_angle * self.P

        # ipdb.set_trace()
        print(f'steering_angle{steering_angle}')
        print(f'heading_error{heading_error}')
        print(f'distance_error{distance_error}')
        print(f'nearest_dist{nearest_dist}')
        self.update_traj(targetWp=self.wp[:2, i], pose=pose)

        return np.clip(steering_angle, -1.0, 1.0), accelation



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
        self.L = 0.8
        self.P = 0.6
        self.planner = PurePursuitPlanner()
        self.odom_subscriber = self.create_subscription(
            Odometry, 'ego_racecar/odom', self.pose_callback, 10)
        drive_topic = '/drive'
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord)  
        self.prev_time = 0
        self.time_thres = 0.02
    
    def pose_callback(self, pose_msg, publish=node_publish, excute=node_execute):
        
        ###### publish waypoints
        if publish:
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
                steering_angle, velocity, interp_point= self.planner.planning(pose=pose, speed=speed, time_interval=time_interval)
                self.prev_time = cur_time
            else:
                return 
        else:
            self.prev_time = cur_time
            return 
        ####### Planning

        ####### publish current target marker
        if publish:
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
            marker.pose.position = Point(x=interp_point[0], y=interp_point[1], z=0.0)
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
