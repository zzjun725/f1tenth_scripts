#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from math import pi, atan2, cos, sin
from queue import PriorityQueue

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# import logging
# # pg.setConfigOptions(leftButtonPan=False)
# logging.basicConfig(level=logging.INFO,
#                     filename='gap_follow_log',
#                     filemode='a')
ROS_HOME='/sim_ws/log/gap_follow'

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: Subscribe to LIDAR
        # TODO: Publish to drive
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord)
        self.scan_subscriber = self.create_subscription(
            LaserScan, lidarscan_topic, self.lidar_callback, 10
        )

        # constant
        self.rb = 20
        self.counter = 0
        self.counter_thres = 5
        self.get_logger().info('Node_start')
        self.safe_thres = 1.0
        self.danger_thres = 0.7          

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = []
        window_size = 5
        for i in range(0, len(ranges), window_size):
            cur_mean = round(sum(ranges[i:i+window_size])/window_size, 6)
            # if cur_mean >= self.safe_thres:
            #     cur_mean = self.safe_thres
            for _ in range(window_size):
                proc_ranges.append(cur_mean)
        proc_ranges = np.array(proc_ranges)
        # print(f'num of safe_points{sum(proc_ranges==self.safe_thres)}')
        # print(np.min(proc_ranges))
        # print(f'len of proc_ranges: {len(proc_ranges)}')
        # print(f'begin of proc_ranges {proc_ranges[0]}')
        # print(f'end of proc_ranges{proc_ranges[-1]}')

        return proc_ranges

    def refine_danger_range(self, start_i, end_i, ranges):
        """
        Return:
        """
        p = start_i
        while p < end_i:
            if ranges[p] <= self.danger_thres:
                ranges[max(0, p-self.rb): p+self.rb] = 0
                p += self.rb
            else:
                p += 1
        return ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        return None
    
    def find_best_point(self, start_i, end_i, closest_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # farmost_p_left = start_i
        # farmost_p_right = start_i
        # farmost_p_range = ranges[start_i]
        # p = start_i
        # while p < end_i:
        #     if ranges[p] >= farmost_p_range:
        #         farmost_p_left = p
        #         farmost_p_range = ranges[p]
        #         p += 1
        #         while p < end_i and ranges[p]==ranges[p-1]:
        #             p += 1
        #         farmost_p_right = p - 1
        #     else:
        #         p += 1
        # if abs(farmost_p_left-closest_i) > abs(farmost_p_right-closest_i):
        #     return farmost_p_left
        # else:
        #     return farmost_p_right
        safe_p_left = start_i
        safe_p_right = end_i
        p = start_i
        safe_range = PriorityQueue()
        while p < end_i:
            if ranges[p] >= self.safe_thres:
                safe_p_left = p
                p+=1
                while p < end_i and ranges[p] >= self.safe_thres:
                    p += 1
                safe_p_right = p-1
                safe_range.put((-(safe_p_right-safe_p_left+1), (safe_p_left, safe_p_right)))
            else:
                p += 1
        if safe_range.empty():
            print('no safe range')
            return np.argmax(ranges)
        else:
            while not safe_range.empty():
                safe_p_left, safe_p_right = safe_range.get()[1]
                target = np.argmax(ranges[safe_p_left:safe_p_right]) + safe_p_left
                # return np.argmax(ranges[safe_p_left:safe_p_right]) + safe_p_left
                if abs(safe_p_left-closest_i) > abs(safe_p_right-closest_i):
                    target = (2*safe_p_left+safe_p_right)//3
                    # return safe_p_left
                else:
                    target = (safe_p_left+2*safe_p_right)//3
                if 179 <= target <= 900:
                    return target
            return target
                # return safe_p_right
        # return (farmost_p_right + farmost_p_left) // 2

    def lidar_callback(self, scan_msg):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        angle_increment = scan_msg.angle_increment
        angle_min = scan_msg.angle_min        
        ranges = scan_msg.ranges

        proc_ranges = self.preprocess_lidar(ranges)
        n = len(proc_ranges)
        # TODO:
        #Find closest point to LiDAR
        closest_p_idx = np.argmin(proc_ranges)
        closest_angle = angle_min + closest_p_idx*angle_increment

        print(f'closest_idx_range {proc_ranges[closest_p_idx]}')
        print(f'closest_p_idx: {closest_p_idx}')
        print(f'closest_angle: {closest_angle}')
        #Eliminate all points inside 'bubble' (set them to zero)
        proc_ranges = self.refine_danger_range(start_i=0, end_i=len(proc_ranges), ranges=proc_ranges)

        #Find max length gap
        #Find the best point in the gap
        # mid = len(proc_ranges) // 2
        # if closest_p_idx <= mid:
        #     farmost_p_idx = self.find_best_point(start_i=closest_p_idx+self.rb, end_i=n-1, 
        #     closest_i = closest_p_idx, ranges=proc_ranges)
        # else:
        #     farmost_p_idx = self.find_best_point(start_i=0, end_i=closest_p_idx-self.rb-1, 
        #     closest_i=closest_p_idx, ranges=proc_ranges)

        farmost_p_idx = self.find_best_point(start_i=0, end_i=len(proc_ranges), closest_i = closest_p_idx, ranges=proc_ranges)
        steering_angle = angle_min + farmost_p_idx*angle_increment
        velocity = 1.0

        print(f'farmost_p_idx: {farmost_p_idx}')
        print(f'farmost_p_range: {proc_ranges[farmost_p_idx]}')
        print(f'steering_angle: {steering_angle*180/pi}')
        if abs(steering_angle) >=30:
            velocity = 0.4
        #Publish Drive message
        if self.counter >= self.counter_thres:
            self.ackermann_ord.drive.speed = velocity
            self.ackermann_ord.drive.steering_angle = steering_angle
            self.drive_publisher.publish(self.ackermann_ord)            
            self.counter = 1
        else:
            self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()