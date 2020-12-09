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

import collections

class Sensor():
	def __init__(self,
		parent_node,
		msg_type,
		topic_name,
		qos_profile = 10,
		):
		self.topic_name = topic_name
		self.parent_node = parent_node
		#self.parent_node.data = None
		self.data = None
		self.parent_node.sub_callback = self.sub_callback
		self.parent_node.subscription = self.parent_node.create_subscription(
			msg_type,
			topic_name,
			self.parent_node.sub_callback,
			qos_profile)

		self.parent_node.get_logger().info(topic_name + ' callback started.')
		self.first_msg = False

	def process_data(self, **kwargs):
		# Main processing should be done here
		raise NotImplementedError  

	def sub_callback(self, msg):
		self.parent_node.get_logger().debug('[%s] Msg received.' %self.topic_name)
		#self.parent_node.get_logger().debug(str(msg))
		#self.parent_node.data = msg
		self.data = msg
		#if not self.first_msg :
				#print(msg.data)
				#self.get_logger().info("First msg received: "+str(msg))
				#self.first_msg = True 

# LIDAR

class LaserScanSensor(Sensor):

	def __init__(self, parent_node):

		super().__init__(
				parent_node = parent_node, 
				msg_type = LaserScan,
				topic_name = "/scan" ,
				qos_profile = qos_profile_sensor_data,
				)

def clean_laserscan(laserscan_data, laser_range = 3.5):
	# Takes only sensed measurements
	for i in range(359):
		if laserscan_data.ranges[i] == float('Inf'):
			laserscan_data.ranges[i] = laser_range #set range to max
		elif np.isnan(laserscan_data.ranges[i]):
			laserscan_data.ranges[i] = 0.0 #set range to 0
		else:
			pass # leave range as it is
	return laserscan_data

def laserscan_2_list(laserscan_data):
	laserscan_list = []
	for index in range(len(laserscan_data.ranges)):
		laserscan_list.append(laserscan_data.ranges[index])
	return laserscan_list #type: list (of float)

def laserscan_2_n_points_list(laserscan_data, n_points = 60):
	n_points_list = []
	len_laserscan_data = len(laserscan_data.ranges)
	for index in range(n_points):
		n_points_list.append(\
			laserscan_data.ranges[int(index*len_laserscan_data/n_points)]
			)
	return n_points_list #type: list (of float)

class CmdVelInfo(Sensor):

	def __init__(self, parent_node):

		super().__init__(
				parent_node = parent_node, 
				msg_type = Twist,
				topic_name = "/cmd_vel" ,
				qos_profile = 10
				)

# ODOMETRY

class OdomSensor(Sensor):

	def __init__(self,
				parent_node,
				msg_type = Odometry,
				topic_name = '/odom',
				qos_profile = 10
				):

		super().__init__(
				parent_node = parent_node,
				msg_type = msg_type,
				topic_name = topic_name,
				qos_profile = qos_profile)

def pose_2_xyyaw(odometry_data):
	# odometry_data:  nav_msgs/msg/Odometry
	pos_x = odometry_data.pose.pose.position.x
	pos_y = odometry_data.pose.pose.position.y
	_,_,yaw = euler_from_quaternion(odometry_data.pose.pose.orientation)
	return pos_x, pos_y, yaw #floats

def euler_from_quaternion(quat):
	"""
	Converts quaternion (w in last place) to euler roll, pitch, yaw
	quat = [x, y, z, w]
	"""
	x = quat.x
	y = quat.y
	z = quat.z
	w = quat.w

	sinr_cosp = 2 * (w*x + y*z)
	cosr_cosp = 1 - 2*(x*x + y*y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2 * (w*y - z*x)
	pitch = np.arcsin(sinp)

	siny_cosp = 2 * (w*z + x*y)
	cosy_cosp = 1 - 2 * (y*y + z*z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw

class s7b3State():
	def __init__(self, parent_node):
		self.parent_node = parent_node

		self.pos_x = collections.deque(maxlen=2)
		self.pos_y = collections.deque(maxlen=2)
		self.yaw =  collections.deque(maxlen=2)
		self.lidar = collections.deque(maxlen=2)

	def initialize_sensors(self):
		self.odom_sensor = OdomSensor(self.parent_node)
		self.laser_scan_sensor = LaserScanSensor(self.parent_node)

	def get_state(self):
		self.lidar.append(laserscan_2_list(
							clean_laserscan(
								self.laser_scan_sensor.data
							)
						))
		x, y, yaw= pose_2_xyyaw(self.odom_sensor.data)
		self.x.append(x), self.y.append(y), self.yaw.append(yaw)

	def get_observation(self):

		pass

	def get_reward(self):

		pass

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

	def update_observation(self):
		processed_lidar = laserscan_2_n_points_list(
			clean_laserscan(
				self.laser_scan_data[-1]
			)
		)
		x, y, yaw = pose_2_xyyaw(self.odometry_data[-1])

		self.observation.append(
			processed_lidar + [x] + [y] + [yaw]
			)

	def compute_reward(self):
		return reward_simple_distance(self.goal_distance, self.goal_angle)

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
