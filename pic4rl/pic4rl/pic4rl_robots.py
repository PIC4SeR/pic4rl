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
from pic4rl.sensors.pic4rl_sensors import clean_laserscan, laserscan_2_list, laserscan_2_n_points_list


import collections

class MobileRobotState():
	def __init__(self):
		# These store callbacks last values
		self.odometry_msg_data = None
		self.laser_scan_msg_data = None

		# These store transitions values
		self.odometry_data = collections.deque(maxlen=2) 
		self.laser_scan_data = collections.deque(maxlen=2)

		self.observation = collections.deque(maxlen=2)

		# Goal a point to reach
		self.goal_position = None
		self.done = None
		self.goal_distance = collections.deque(maxlen=2)
		self.goal_angle = collections.deque(maxlen=2)


	def update_state(self):
		self.odometry_data.append(self.odometry_msg)
		self.laser_scan_data.append(self.laser_scan_msg)
		x, y, yaw = pose_2_xyyaw(self.odometry_data[-1])

		self.goal_distance.append(
			goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y))
		self.goal_angle.append(
			goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y))
		self.check_done()

	def update_observation(self):
		processed_lidar = laserscan_2_n_points_list(
			clean_laserscan(
				self.laser_scan_data[-1]
			)
		)
		x, y, yaw = pose_2_xyyaw(self.odometry_data[-1])

		self.observation.append(
		#	processed_lidar +
		#	[x] +
		#	[y] +
			[self.goal_distance[-1]] + 
			[yaw]
			)

	def compute_reward(self):

		self.reward = reward_simple_distance(self.goal_distance, self.goal_angle)

	def check_done(self):
		self.done = False
		min_collision_range = 0.25 # m
		for measure in laserscan_2_list(
				clean_laserscan(self.laser_scan_msg)):
			if 0.05 < measure < min_collision_range:
				self.get_logger().info('Collision!!')
				self.done = True
				return 
		if self.goal_distance[-1] < 0.1:
			self.get_logger().info('GOAL REACHED')
			self.done = True
			return
			

def reward_simple_distance(goal_distance,goal_angle):
	# This function expects history related to goal
	return (goal_distance[0]-goal_distance[1])

def goal_pose_to_distance(pos_x, pos_y, goal_pos_x, goal_pos_y):
	return math.sqrt((goal_pos_x-pos_x)**2
			+ (goal_pos_y-pos_y)**2)

def goal_pose_to_angle(pos_x, pos_y, yaw,goal_pos_x, goal_pos_y):
		path_theta = math.atan2(
			goal_pos_y-pos_y,
			goal_pos_y-pos_x)

		goal_angle = path_theta - yaw

		if goal_angle > math.pi:
			goal_angle -= 2 * math.pi

		elif goal_angle < -math.pi:
			goal_angle += 2 * math.pi
		return goal_angle


