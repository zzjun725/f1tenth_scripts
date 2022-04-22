#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from math import pi, atan2, cos, sin
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import time

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers
        self.ackermann_ord = AckermannDriveStamped()
        self.ackermann_ord.drive.acceleration = 0.        
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.drive_publisher.publish(self.ackermann_ord)
        self.scan_subscriber = self.create_subscription(
            LaserScan, lidarscan_topic, self.scan_callback, 10
        )        
        # TODO: set PID gains
        self.kp = 16
        self.kd = 0.2
        # self.kd = 0
        self.ki = 0

        # TODO: store history
        self.integral = 0
        self.integral_thres = 8
        self.prev_error = 0.0
        self.error = 0
        self.targetD = 0.7
        self.lookforwardD = 1.0
        # self.time_interval = 0.02
        self.prev_time = 0
        self.time_thres = 0.02
        # self.counter = 0
        # self.counter_thres = 2

        # TODO: store any necessary values you think you'll need
  
    
    def pid_control(self, error, velocity, time_interval):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        P_term = self.kp*error

        self.integral += error*time_interval
        I_term = min(self.integral_thres, self.ki*self.integral)

        D_term = self.kd*(error - self.prev_error) / time_interval
        self.prev_error = error

        # TODO: Use kp, ki & kd to implement a PID controller
        steering_angle = P_term + I_term + D_term
        print(f'P_term{P_term}  ', f'I_term{I_term}  ', f'D_term{D_term}  ')
        print(f'steering_angle: {steering_angle}')
        if 0 <= abs(steering_angle) <= 10:
            velocity = 3.5
        elif 10 < abs(steering_angle) < 20:
            velocity = 1.0
        # if abs(steering_angle) <= 0.5:
        #     steering_angle = 0.0
        # print(f'velocity: {velocity}')
        # TODO: fill in drive message and publish
        # velocity = 0.05
        self.ackermann_ord.drive.speed = velocity
        self.ackermann_ord.drive.steering_angle = steering_angle * pi / 180
        self.drive_publisher.publish(self.ackermann_ord)

    def scan_callback(self, scan_msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        # import ipdb; ipdb.set_trace()
        angle_increment = scan_msg.angle_increment
        angle_min = scan_msg.angle_min
        distances = scan_msg.ranges 

        theta = pi/6
        b = distances[int((pi/2 - angle_min) // angle_increment)]
        a = distances[int((pi/2 + theta - angle_min) // angle_increment)]
        alpha = atan2(a*cos(theta)-b, a*sin(theta))
        D = b*cos(alpha)
        error = (D-self.lookforwardD*sin(alpha)) - self.targetD
        if abs(error - self.prev_error) < 1e-3:
            return 
        # error = -error
        # print(f'D:{D}', f'lookforwardD:{self.lookforwardD*sin(alpha)}')
        # PID
        velocity = 2.0 # TODO: calculate desired car velocity based on error
        # if self.counter >= self.counter_thres:
        cur_time = scan_msg.header.stamp.nanosec/1e9 + scan_msg.header.stamp.sec
        time_interval = cur_time - self.prev_time
        if self.prev_time != 0.0:
            if time_interval > self.time_thres:
                self.pid_control(error, velocity, time_interval) # TODO: actuate the car with PID
                self.prev_time = cur_time
                # pass
        else:
            self.prev_time = cur_time
        print(time_interval)
        # time.sleep(3)
        #     self.counter = 1
        # else:
        #     self.counter += 1
        


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()