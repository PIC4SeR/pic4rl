#!/usr/bin/env python3
#
# MIT License

# Copyright (c) 2021 PIC4SeR
# Authors: Enrico Sutera (enricosutera), Mauro Martini(maurom3197)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# General purpose
import time
import numpy as np
import math

import gym
from gym import spaces

from pic4rl.sensors.pic4rl_sensors import pose_2_xyyaw
from pic4rl.sensors.pic4rl_sensors import clean_laserscan, laserscan_2_list, laserscan_2_n_points_list

import collections

class OdomGoalLidarCollision():
	"""
	This class allows a 
	"""
	def __init__(self,
				collision_distance = 0.2,
				min_distance = 0.15,
				episode_timeout = 20 #seconds
				):
		self.get_logger().info('[OdomGoalLidarCollision] Initialization.')
		self.collision_distance = collision_distance
		self.min_distance = min_distance
		self.episode_timeout = episode_timeout
		self.episode_start_time = time.time()

	def check_done(self):
		self.done = False
		# Check collision
		min_collision_range = 0.25 # m
		for measure in laserscan_2_list(
				clean_laserscan(self.laser_scan_msgs[-1])):
			if 0.05 < measure < self.collision_distance:
				self.get_logger().info('[OdomGoalLidarCollision] Collision!!')
				self.done = True
				return 

		# Check goal
		x, y, _ = pose_2_xyyaw(self.odometry_msgs[-1])
		goal_distance = goal_pose_to_distance(x,y,self.goal_pos_x, self.goal_pos_y)
		
		if goal_distance < self.min_distance:
			self.get_logger().info('[OdomGoalLidarCollision] Goal!!')
			self.done = True
			return

		if time.time() - self.episode_start_time > self.episode_timeout:
			self.get_logger().info('[OdomGoalLidarCollision] Timeout!!')
			self.done = True
			self.episode_start_time = time.time() # Set current time as start
			# This actually is incorrect, since some times passes before episode start
			return

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
