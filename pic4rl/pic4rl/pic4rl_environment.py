#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert

import os
import random
import sys
import time

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math 


from rclpy.qos import QoSProfile

class Pic4rlEnvironment(Node):
	def __init__(self):
		super().__init__('pic4rl_environment')
		# To see debug logs
		#rclpy.logging.set_logger_level('omnirob_rl_environment', 10)

		"""************************************************************
		** Initialise ROS publishers and subscribers
		************************************************************"""
		qos = QoSProfile(depth=10)

		self.cmd_vel_pub = self.create_publisher(
			Twist,
			'cmd_vel',
			qos)

		# Initialise client
		#self.send_twist = self.create_client(Twist, 'send_twist')

		#self.task_succeed_client = self.create_client(Empty, 'task_succeed')
		#self.task_fail_client = self.create_client(Empty, 'task_fail')

		self.pause_physics_client = self.create_client(Empty, 'pause_physics')
		self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

		self.get_state_client = self.create_client(State, 'get_state')
		self.new_episode_client = self.create_client(Reset, 'new_episode')

		"""##########
		State variables
		##########"""
		self.init_step = True
		self.episode_step = 0
		self.goal_pos_x = None
		self.goal_pos_y = None
		self.previous_twist = None
		self.previous_pose = Odometry()

		#test variable
		self.step_flag = False
		self.twist_received = None


		"""##########
		Environment initialization
		##########"""

	"""#############
	Main functions
	#############"""

	def render(self):

		pass

	def step(self, action):
		twist = Twist()
		twist.linear.x = float(action[0])
		twist.linear.y = float(action[1])
		twist.angular.z = float(action[2])
		observation, reward, done = self._step(twist)
		info = None
		return observation, reward, done, info

	def _step(self, twist=Twist(), reset_step = False):
		#After environment reset sensors data are not instaneously available
		#that's why there's the while. A timer could be added to increase robustness
		data_received = False
		while not data_received:
			# Send action
			self.send_action(twist)
			# Get state
			state = self.get_state()
			data_received = state.data_received



		lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_state(state)

		# Check events (failure,timeout, success)
		done, event = self.check_events(lidar_measurements, goal_distance, self.episode_step)

		if not reset_step:
			# Get reward
			reward = self.get_reward(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)
			observation = self.get_observation(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw)
		else:
			reward = None
			observation = None

		# Send observation and reward
		self.update_state(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)

		return  observation, reward, done

	def reset(self):
		#self.destroy_subscription('cmd_vel')
		req = Reset.Request()
		req.goal_pos_x,req.goal_pos_y = self.get_goal()
		self.get_logger().info("Environment reset ...")

		while not self.new_episode_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		future = self.new_episode_client.call_async(req)
		#self.get_logger().debug("Reset env request sent ...")
		#rclpy.spin_until_future_complete(self, future,timeout_sec=1.0)
		#time_start = time.time()
		while rclpy.ok():
			rclpy.spin_once(self,timeout_sec=2)
			if future.done():
				if future.result() is not None:
					self.get_logger().debug("Environment reset done")
					break 
			#if  time.time() - time_start > 10:
			#	raise ValueError("In realtà non è un ValueError")
			
	
		self.get_logger().debug("Performing null step to reset variables")
		_,_,_, = self._step(reset_step = True)
		observation,_,_, = self._step()
		return observation

	"""#############
	Secondary functions (used in main functions)
	#############"""

	def send_action(self,twist):
		self.get_logger().debug("unpausing...")
		self.unpause()
		self.get_logger().debug("publishing twist...")
		self.cmd_vel_pub.publish(twist)
		time.sleep(0.1)
		self.get_logger().debug("pausing...")
		self.pause()	

	def get_state(self):
		self.get_logger().debug("Asking for the state...")
		req = State.Request()
		future =self.get_state_client.call_async(req)
		rclpy.spin_until_future_complete(self, future)
		try:
			state = future.result()
		except Exception as e:
			node.get_logger().error('Service call failed %r' % (e,))
		self.get_logger().debug("State received ...")
		return state

	def process_state(self,state):

		self.episode_step += 1

		#from LaserScan msg to 359 len filterd list
		lidar_measurements = self.filter_laserscan(state.scan)

		#from Odometry msg to x,y, yaw, distance, angle wrt goal
		goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_odom(state.odom)

		return lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw

	def check_events(self, lidar_measurements, goal_distance, step):

		min_range = 0.26

		if  0.05 <min(lidar_measurements) < min_range:
			# Collision
			self.get_logger().info('Collision')
			return True, "collision"

		if goal_distance < 0.2:
			# Goal reached
			self.get_logger().info('Goal')
			return True, "goal"

		if step >= 500:
			#Timeout
			self.get_logger().info('Timeout')
			return True, "timeout"

		return False, "None"

	def get_observation(self, twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw):

		return np.array([goal_distance, goal_angle, yaw])

	def get_reward(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
		reward = self.previous_goal_distance - goal_distance
		if event == "goal":
			reward+=10
		self.get_logger().debug(str(reward))
		return reward

	def get_goal(self):
		x = random.uniform(-3,3)
		y = random.uniform(-3,3)
		self.get_logger().info("New goal")
		#self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
		self.goal_pose_x = x
		self.goal_pose_y = y
		return x,y

	def update_state(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):
		#Here state variables are updated
		self.episode_step += 1
		self.previous_twist = twist
		self.previous_lidar_measurements = lidar_measurements
		self.previous_goal_distance = goal_distance
		self.previous_goal_angle = goal_angle
		self.previous_pos_x = pos_x
		self.previous_pos_y = pos_y
		self.previous_yaw = yaw
		# If done, set flag for resetting everything at next step
		if done:
			self.init_step = True
			self.episode_step = 0

	"""#############
	Auxiliar functions (used in secondary functions)
	#############"""

	def pause(self):
		req = Empty.Request()
		while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		self.pause_physics_client.call_async(req) 

	def unpause(self):
		req = Empty.Request()
		while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		self.unpause_physics_client.call_async(req) 

	def filter_laserscan(self,laserscan_msg):
		# There are some outliers (0 or nan values, they all are set to 0) that will not be passed to the DRL agent

		# Correct data:
		scan_range = []

		# Takes only sensed measurements
		for i in range(359):
			if laserscan_msg.ranges[i] == float('Inf'):
				scan_range.append(3.50)
			elif np.isnan(laserscan_msg.ranges[i]):
				scan_range.append(0.00)
			else:
				scan_range.append(laserscan_msg.ranges[i])
		return scan_range

	def process_laserscan(self,laserscan_msg):

		pass

	def process_odom(self, odom_msg):
		#self.previous_pose.pose.pose.position.x = odom_msg.pose.pose.position.x
		#self.previous_pose.pose.pose.position.y = odom_msg.pose.pose.position.y

		pos_x = odom_msg.pose.pose.position.x
		pos_y = odom_msg.pose.pose.position.y
		_,_,yaw = self.euler_from_quaternion(odom_msg.pose.pose.orientation)

		goal_distance = math.sqrt(
			(self.goal_pose_x-pos_x)**2
			+ (self.goal_pose_y-pos_y)**2)

		path_theta = math.atan2(
			self.goal_pose_y-pos_y,
			self.goal_pose_x-pos_x)

		goal_angle = path_theta - yaw

		if goal_angle > math.pi:
			goal_angle -= 2 * math.pi

		elif goal_angle < -math.pi:
			goal_angle += 2 * math.pi

		self.goal_distance = goal_distance
		self.goal_angle = goal_angle

		return goal_distance, goal_angle, pos_x, pos_y, yaw

	def euler_from_quaternion(self, quat):
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



def main(args=None):
	rclpy.init()
	omnirob_rl_environment = OmnirobRlEnvironment()
	omnirob_rl_environment.spin()

	omnirob_rl_environment.get_logger().info('Node spinning ...')
	rclpy.spin_once(omnirob_rl_environment)

	omnirob_rl_environment.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
