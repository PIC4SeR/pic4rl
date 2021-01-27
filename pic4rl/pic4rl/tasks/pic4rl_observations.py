#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import random 
import math 
import os
from ament_index_python.packages import get_package_share_directory

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

from pic4rl.sensors.pic4rl_sensors import pose_2_xyyaw
from pic4rl.sensors.pic4rl_sensors import clean_laserscan, laserscan_2_list, laserscan_2_n_points_list

import gym
from gym import spaces


import collections


class OdomLidarObs():

	"""
	This Observation class is compatible with 
		OdomLidarState
	"""
	def __init__(self,\
				lidar_points = 60):
		self.get_logger().debug('[OdomLidarObs] Initialization.')
		self.lidar_points = lidar_points

		self.observation = collections.deque(maxlen=2) 

	def update_observation(self):
		processed_lidar = laserscan_2_n_points_list(
			clean_laserscan(self.laser_scan_msgs[-1]),\
			self.lidar_points
		)
		x, y, yaw = pose_2_xyyaw(self.odometry_msgs[-1])
		goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
		goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
		self.observation.append([goal_distance] + \
								[goal_angle] 	+ \
								processed_lidar)

class OdomObs():


	"""
	This Observation class is compatible with 
		OdomLidarState
	"""
	def __init__(self):
		self.get_logger().debug('[OdomObs] Initialization.')

		self.observation = collections.deque(maxlen=2) 

	def update_observation(self):
		x, y, yaw = pose_2_xyyaw(self.odometry_msgs[-1])
		goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
		goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
		self.observation.append( [goal_distance]+[goal_angle])

class OdomDepthObs():
	def __init__(self):
		self.get_logger().debug('[OdomDepthObs] Initialization.')

		self.observation = collections.deque(maxlen=2) 

	def update_observation(self):
		x, y, yaw = pose_2_xyyaw(self.odometry_msgs[-1])
		goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
		goal_angle = goal_pose_to_angle(x,y,yaw, self.goal_pos_x, self.goal_pos_y)
		depth_img = clean_raw_depth(self.generic_depth_camera_img)
		self.observation.append([goal_distance] + \
								[goal_angle] 	+ \
								depth_img)


"""
# Auxiliar functions ()
"""

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

