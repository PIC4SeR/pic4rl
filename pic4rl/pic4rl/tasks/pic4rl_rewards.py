#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import random 
import math 

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

from pic4rl.sensors.pic4rl_sensors import pose_2_xyyaw
from pic4rl.tasks.pic4rl_state import goal_pose_to_distance, goal_pose_to_angle

import collections

class SimpleDistanceReward():
	"""
	This class is compatible with:
		OdomLidarState
	"""
	def __init__(self):
		self.get_logger().debug('[SimpleDistanceReward] Initialization.')
		self.reward = None

	def compute_reward(self):
		x_0, y_0, yaw_0 = pose_2_xyyaw(self.odometry_msgs[-2]) # previous step
		x_1, y_1, yaw_1 = pose_2_xyyaw(self.odometry_msgs[-1]) # current  step
		goal_distance_0 = goal_pose_to_distance( # previous step
							x_0,
							y_0,
							self.goal_pos_x,
							self.goal_pos_y)
		goal_distance_1 = goal_pose_to_distance( # current  step
							x_1,
							y_1,
							self.goal_pos_x,
							self.goal_pos_y)

		#goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
		self.reward = (goal_distance_0 - goal_distance_1)
