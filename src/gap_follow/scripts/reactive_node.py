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
        self.rb = 10
        self.counter = 0
        self.counter_thres = 2
        self.get_logger().info('Node_start')
        self.safe_thres = 2.0
        self.danger_thres = 1.0
        self.cut_thres = 2.0
        self.max_speed = 4.0
        self.min_speed = 3.0
        self.max_gap = 300
        self.min_gap = 50
        self.drive = True
        self.window_size = 20

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        proc_ranges = []
        window_size = self.window_size
        for i in range(0, len(ranges), window_size):
            cur_mean = round(sum(ranges[i:i+window_size])/window_size, 5)
            # if cur_mean >= self.safe_thres:
            #     cur_mean = self.safe_thres
            for _ in range(window_size):
                proc_ranges.append(cur_mean)
        proc_ranges = np.array(proc_ranges)
        # print(f'num of safe_points{sum(proc_ranges==self.safe_thres)}')

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
    
    def find_best_point(self, start_i, end_i, closest_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        safe_p_left = start_i
        safe_p_right = end_i
        p = start_i
        safe_range = PriorityQueue()
        while p < end_i:
            if ranges[p] >= self.safe_thres:
                safe_p_left = p
                p+=1
                # while p < end_i and ranges[p] >= self.safe_thres and p-safe_p_left <= 290:
                while p < end_i and ranges[p] >= self.safe_thres and p-safe_p_left <= self.max_gap and ranges[p] - ranges[max(0, p-1)] < self.cut_thres:
                    p += 1
                safe_p_right = max(0, p-1)
                if safe_p_right != safe_p_left:
                    # try:
                    safe_range.put((-(np.max(ranges[safe_p_left:safe_p_right])), (safe_p_left, safe_p_right)))
                    # except:
                        # import ipdb; ipdb.set_trace()
                        # continue
            else:
                p += 1
        if safe_range.empty():
            print('no safe range')
            return np.argmax(ranges)
        else:
            while not safe_range.empty():
                safe_p_left, safe_p_right = safe_range.get()[1]
                target = (safe_p_left+safe_p_right)//2
                if 179 <= target <= 900 and safe_p_right-safe_p_left > self.min_gap:
                    print(f'left: {safe_p_left}, right: {safe_p_right}')
                    return target
            return target

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

        # print(f'closest_idx_range {proc_ranges[closest_p_idx]}')
        # print(f'closest_p_idx: {closest_p_idx}')
        # print(f'closest_angle: {closest_angle}')
        #Eliminate all points inside 'bubble' (set them to zero)
        proc_ranges = self.refine_danger_range(start_i=0, end_i=len(proc_ranges), ranges=proc_ranges)

        farmost_p_idx = self.find_best_point(start_i=0, end_i=len(proc_ranges), closest_i = closest_p_idx, ranges=proc_ranges)
        steering_angle = angle_min + farmost_p_idx*angle_increment
        # velocity = 6.5
        velocity = self.max_speed

        print(f'farmost_p_idx: {farmost_p_idx}')
        print(f'farmost_p_range: {proc_ranges[farmost_p_idx]}')
        print(f'steering_angle: {steering_angle*180/pi}')
        if abs(steering_angle) >=0.3:
            velocity = self.min_speed
        #Publish Drive message
        self.ackermann_ord.drive.speed = velocity
        self.ackermann_ord.drive.steering_angle = steering_angle

        if self.drive:
            self.drive_publisher.publish(self.ackermann_ord)            


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

