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

import gym
from gym import spaces


import collections

class WheeledRobot():
	def __init__(self):

		max_linear_x_speed = 0.3
		min_linear_x_speed = -0.3

		#max_linear_y_speed = 0.5
		#min_linear_y_speed = -0.5

		max_angular_z_speed = 1
		min_angular_z_speed = -1.0

		action =[
			[min_linear_x_speed, max_linear_x_speed],
			[min_angular_z_speed, max_angular_z_speed]
			#[-0.5, 0.5], # x_speed 
			##[-0.5, 0.5], # y_speed
			#[-1, 1] # theta_speed
		]


		low_action = []
		high_action = []
		for i in range(len(action)):
			low_action.append(action[i][0])
			high_action.append(action[i][1])

		low_action = np.array(low_action, dtype=np.float32)
		high_action = np.array(high_action, dtype=np.float32)

		self.action_space = spaces.Box(
			low=low_action,
			high=high_action,
			#shape=(1,),
			dtype=np.float32
		)
		
		"""
		state
		"""
		state =[

		[0., 5.], # goal_distance 
		[-math.pi, math.pi], # goal_angle
		#[-math.pi, math.pi] # yaw
		]
		

		low_state = []
		high_state = []
		for i in range(len(state)):
			low_state.append(state[i][0])
			high_state.append(state[i][1])

		self.low_state = np.array(low_state, dtype=np.float32)
		self.high_state = np.array(high_state, dtype=np.float32)

		self.observation_space = spaces.Box(
			low=self.low_state,
			high=self.high_state,
			dtype=np.float32
		)

#class DifferentialRobot(WheeledRobot):

"""
# STATES
"""

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

"""
# OBSERVATIONS
"""


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

"""
# GOALS
"""

class RandomGoal():
	"""
	Generate random goal coorditanes x,y given a range (min, max)
	and store them in 
		self.goal_pos_x
		self.goal_pos_y
	"""
	def __init__(self, goal_range = (-3,3)):
		self.get_logger().info('[RandomGoal] Initialization. ')
		self._goal_range = goal_range
		self.goal_pos_x = None
		self.goal_pos_y = None

	def new_goal(self):

		self.goal_pos_x = random.uniform(
			self._goal_range[0],self._goal_range[1])
		self.goal_pos_y = random.uniform(
			self._goal_range[0],self._goal_range[1])
		msg = '[RandomGoal] New goal x, y : {:.2f}, {:.2f}'.format(self.goal_pos_x, self.goal_pos_y)
		self.get_logger().info(msg)

"""
# ENDS OF EPISODE
"""

class OdomGoalLidarCollision():
	"""
	This class allows a 
	"""
	def __init__(self,
	 			collision_distance = 0.2,
	 			min_distance = 0.15
				):
		self.get_logger().info('[OdomGoalLidarCollision] Initialization.')
		self.collision_distance = collision_distance
		self.min_distance = min_distance

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

