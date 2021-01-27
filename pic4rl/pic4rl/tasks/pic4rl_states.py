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


class OdomLidarState():
	# Sensors callbacks store msgs in this attributes:
	# 	>> self.laser_scan_msg_data <<
	# 	>> self.odometry_msg_data <<
	def __init__(self):
		self.get_logger().debug('[OdomLidarState] Initialization.')

		self.odometry_msgs = collections.deque(maxlen=2) 
		self.laser_scan_msgs = collections.deque(maxlen=2) 
		self.goal_pose = collections.deque(maxlen=2) 
		self.done = None

	def update_state(self):
		self.odometry_msgs.append(self.odometry_msg)
		self.laser_scan_msgs.append(self.laser_scan_msg)

class OdomState():
	# Sensors callbacks store msgs in this attributes:
	# 	>> self.odometry_msg_data <<

	def __init__(self):
		self.get_logger().debug('[OdomState] Initialization.')

		self.odometry_msgs = collections.deque(maxlen=2) 
		self.goal_pose = collections.deque(maxlen=2) 
		self.done = None

	def update_state(self):
		self.odometry_msgs.append(self.odometry_msg)

class OdomDepthState():
	# Sensors callbacks store msgs in this attributes:
	# 	>> self.laser_scan_msg_data <<
	# 	>> self.odometry_msg_data <<
	def __init__(self):
		self.get_logger().debug('[OdomDepthState] Initialization.')

		self.odometry_msgs = collections.deque(maxlen=2) 
		self.camera_depth_msgs = collections.deque(maxlen=2) 
		self.done = None

	def update_state(self):
		self.odometry_msgs.append(self.odometry_msg)
		self.camera_depth_msgs.append(self.camera_depth_msg)


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

