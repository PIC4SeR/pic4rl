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
from pic4rl.sensors.pic4rl_sensors_class import Sensors

from pic4rl.tasks.pic4rl_state import WheeledRobot
from pic4rl.tasks.pic4rl_state import OdomLidarState, OdomLidarObs
from pic4rl.tasks.pic4rl_state import OdomState, OdomObs
from pic4rl.tasks.pic4rl_state import RandomGoal
from pic4rl.tasks.pic4rl_state import OdomGoalLidarCollision
from pic4rl.tasks.pic4rl_state import *

from pic4rl.tasks.pic4rl_rewards import SimpleDistanceReward

import gym
from gym import spaces


import collections

class Pic4Navigation():
	def __init__(self):
		pass
		# Robot
			# Sensors
		# Reward 
		# State
		# 
class LidarNavigation(
	OdomGoalLidarCollision,
	RandomGoal, 
	SimpleDistanceReward,
	OdomLidarObs,
	OdomLidarState,
	Sensors,
	WheeledRobot):

	# This class refers to a navigation task in which
	# a ground mobile robot uses
	#	Odometry 	(<-- /odom)
	#	Lidar 		(<-- /scan)
	# 
	# To reach goals that are spawned.
	# The robot can have differential/holonomic locomotion
	# system
	#
	#
	#
	def __init__(self):
		self.get_logger().debug('[LidarNavigation] Initialization.')
		Sensors.__init__(self, 
				generic_laser_scan_sensor = True,
				odometry_sensor = True)
		WheeledRobot.__init__(self) #currently only differential
		OdomLidarState.__init__(self)
		OdomLidarObs.__init__(self)
		SimpleDistanceReward.__init__(self)
		RandomGoal.__init__(self)
		OdomGoalLidarCollision.__init__(self)
class OdomNavigation(
	SimpleDistanceReward,
	OdomObs,
	OdomState,
	Sensors,
	WheeledRobot):

	# This class refers to a navigation task in which
	# a ground mobile robot uses
	#	Odometry 	(<-- /odom)
	#	Lidar 		(<-- /scan)
	# 
	# To reach goals that are spawned.
	# The robot can have differential/holonomic locomotion
	# system
	#
	#
	#
	def __init__(self):
		self.get_logger().debug('[OdomNavigation] Initialization.')
		Sensors.__init__(self, 
				generic_laser_scan_sensor = True,
				odometry_sensor = True)
		WheeledRobot.__init__(self) #currently only differential
		OdomState.__init__(self)
		OdomObs.__init__(self)
		SimpleDistanceReward.__init__(self)
class OdomDepthNavigation(
	OdomGoalLidarCollision,
	RandomGoal, 
	SimpleDistanceReward,
	OdomDepthObs,
	OdomDepthState,
	Sensors,
	WheeledRobot):

	# This class refers to a navigation task in which
	# a ground mobile robot uses
	#	Odometry 	(<-- /odom)
	#	Lidar 		(<-- /scan)
	# 
	# To reach goals that are spawned.
	# The robot can have differential/holonomic locomotion
	# system
	#
	#
	#
	def __init__(self):
		self.get_logger().debug('[LidarNavigation] Initialization.')
		Sensors.__init__(self, 
				generic_laser_scan_sensor = True,
				odometry_sensor = True)
		WheeledRobot.__init__(self) #currently only differential
		OdomLidarState.__init__(self)
		OdomLidarObs.__init__(self)
		SimpleDistanceReward.__init__(self)
		RandomGoal.__init__(self)
		OdomGoalLidarCollision.__init__(self)








#class DifferentialRobot(WheeledRobot):

