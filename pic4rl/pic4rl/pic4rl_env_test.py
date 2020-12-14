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
import tensorflow as tf

import random
import sys
import time

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
import numpy as np
import math

from numpy import savetxt
import cv2
from cv_bridge import CvBridge


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

		self.Image_sub = self.create_subscription(
		 Image,
		'/camera/depth/image_raw',
		self.DEPTH_callback,
		qos_profile=qos_profile_sensor_data)

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
		self.goal_index = 0
		self.previous_twist = None
		self.previous_pose = Odometry()
		self.previous_pos = Odometry()
		self.total_distance = 0
		self.goal_distance = 0

		self.stage = 1
		self.lidar_points = 359
		self.cutoff = 5
		#self.depth_image = np.zeros((240,320), np.uint8)
		self.bridge = CvBridge()		
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
		#twist.linear.y = float(action[1])
		twist.angular.z = float(action[1])
		observation, reward, done = self._step(twist)
		info = None
		return observation, reward, done, info, self.total_distance, self.goal_distance

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

		lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image = self.process_state(state, reset_step)

		# Check events (failure,timeout, success)
		done, event = self.check_events(lidar_measurements, goal_distance, self.episode_step)

		if not reset_step:
			# Get reward
			reward = self.get_reward(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)
			observation = self.get_observation(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image)
		else:
			reward = None
			observation = None
			self.path_length = 0

		# Send observation and reward
		self.update_state(twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event)

		return  observation, reward, done

	def reset(self, episode):
		#self.destroy_subscription('cmd_vel')
		self.episode = episode
		req = Reset.Request()
		req.goal_pos_x,req.goal_pos_y = self.get_goal(episode)
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
		return observation, self.goal_pose_x, self.goal_pose_y

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

	def process_state(self,state, reset_step):

		self.episode_step += 1

		#from LaserScan msg to 359 len filterd list
		lidar_measurements = self.filter_laserscan(state.scan)
		#from 359 filtered lidar points to 60 selected lidar points
		lidar_measurements = self.process_laserscan(lidar_measurements)

		#from Odometry msg to x,y, yaw, distance, angle wrt goal
		goal_distance, goal_angle, pos_x, pos_y, yaw = self.process_odom(state.odom, reset_step)

		#process Depth Image from sensor msg
		depth_image = self.process_depth_image()

		return lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, depth_image

	def check_events(self, lidar_measurements, goal_distance, step):

		min_range = 0.25

		if  0.05 <min(lidar_measurements) <= min_range:
			# Collision
			self.get_logger().info('Collision')
			return True, "collision"

		if goal_distance < 0.2:
			# Goal reached
			self.get_logger().info('Goal')
			return True, "goal"

		if step >= 1000:
			#Timeout
			self.get_logger().info('Timeout')
			return True, "timeout"

		return False, "None"

	def get_observation(self, twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw,depth_image):
		
		#WITH DEPTH CAMERA
		# state_list = []
		# goal_dist = goal_distance/self.cutoff
		# goal_angle_norm = goal_angle/(math.pi)
		# goal_info = np.array([goal_dist, goal_angle_norm], dtype=np.float32)
		# goal_info =tf.convert_to_tensor(goal_info)
		# state_list.append((goal_info))
		# state_list.append(depth_image)

		# return state_list

		#WITH LIDAR
		state_list = []
		state_list.append(float(goal_distance))
		state_list.append(float(goal_angle))

		#state_list.append(float(self.min_obstacle_distance))
		#state_list.append(float(self.min_obstacle_angle))
		for point in lidar_measurements:
			state_list.append(float(point))
			#print(point)
		state = np.array(state_list,dtype = np.float32)
		return state

		
	def get_reward(self,twist,lidar_measurements, goal_distance, goal_angle, pos_x, pos_y, yaw, done, event):   
		reward = 0

		if event == "goal":
			reward += 100
		elif event == "collision":
			reward += -50
		elif event == "timeout":
			reward += -25
		self.get_logger().debug(str(reward))

		#print("Score:", reward)
		return reward

	def get_goal(self,episode):
		goal_pose_list_easy = [[2.0, -1.5],[1.2, -1.8],[0.2, -2.0], [2.0, 2.0], [0.8, 2.0],
						[-1.9, 1.2], [-1.9, -0.5], [-2.0, -2.0]]
		goal_pose_list = [[3.0, 2.0],  [-3.0, -2.0], [-0.2, 4.0],  [-2.0, -4.0], [-4.0, 1.0], [-2.5, -2.5], [2.2, 4.0], [3.5, 4.0], [2.5, -4.4],[4.5,4.5], [-4.2, -4.2],[3.6, 3.6], 
						[1.0, -4.0], [-1.9, -4.0], [-4.5, -3.0], [-4.1, 4.1],  [2.3, 4.2],  [-2.4, 4.2],  [1.3, -4.2],[-4.4, -1.0],  [4.0, 2.5],[-4.5, 0.8],[-0.5, -4.2], [-4.1, 0.0]]

		x = goal_pose_list[self.goal_index][0]
		y = goal_pose_list[self.goal_index][1]
		self.goal_index += 1
		print("Goal pose: ", x, y)

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
		for i in range(self.lidar_points):
			if laserscan_msg.ranges[i] == float('Inf'):
				scan_range.append(3.50)
			elif np.isnan(laserscan_msg.ranges[i]):
				scan_range.append(0.00)
			else:
				scan_range.append(laserscan_msg.ranges[i])

		return scan_range

	def process_laserscan(self,lidar_pointlist):

		scan_range_process = []
		min_dist_point = 100
		# Takes only 60 lidar points
		for i in range(self.lidar_points):
			if lidar_pointlist[i] < min_dist_point:
				min_dist_point = lidar_pointlist[i]
			if i % 10 == 0:
				scan_range_process.append(min_dist_point)
				min_dist_point = 100
		#print('selected lidar points:', len(scan_range_process))

		self.min_obstacle_distance = min(scan_range_process)
		self.min_obstacle_angle = np.argmin(scan_range_process)

		return scan_range_process
 
	def DEPTH_callback(self, msg):
		depth_image_raw = np.zeros((120,160), np.uint8)
		depth_image_raw = self.bridge.imgmsg_to_cv2(msg, '32FC1')
		self.depth_image_raw = np.array(depth_image_raw, dtype= np.float32)
		#print(self.depth_image_raw.shape)
		#savetxt('/home/mauromartini/mauro_ws/depth_images/text_depth_image_raw.csv', depth_image_raw, delimiter=',')
		#np.save('/home/maurom/depth_images/depth_image.npy', depth_image_raw)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_raw.png', self.depth_image_raw)

    #@tf.function
	def process_depth_image(self):
		img = np.array(self.depth_image_raw, dtype= np.float32)
		#savetxt('/home/maurom/depth_images/text_depth_image.csv', depth_image, delimiter=',')
		#print('image shape: ', depth_image.shape)

		#check crop is performed correctly
		#img = tf.convert_to_tensor(self.depth_image_raw, dtype=tf.float32)
		#img = img.reshape(240,320,1)
		img = tf.reshape(img, [120,160,1])
		#width =304
		#height = 228
		#h_off = int((240-height)*0.5)
		#w_off = int((320-width)*0.5)
		#img_crop = tf.image.crop_to_bounding_box(img,h_off,w_off,height,width)
		img_resize = tf.image.resize(img,[60,80])
		depth_image = tf.reshape(img_resize, [60,80])
		depth_image = np.array(depth_image, dtype= np.float32)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_res.png', depth_image)
		#savetxt('/home/mauromartini/mauro_ws/depth_images/depth_image_60_80.csv', depth_image, delimiter=',')
		depth_image = self.depth_rescale(depth_image, self.cutoff)
		#print(depth_image.shape)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_cutoff.png', depth_image)
		#savetxt('/home/mauromartini/mauro_ws/depth_images/depth_image_cutoff.csv', depth_image, delimiter=',')

		self.image_size = depth_image.shape
		return depth_image

	def depth_rescale(self,img, cutoff):
		#Useful to turn the background into black into the depth images.
		w,h = img.shape
		#new_img = np.zeros([w,h,3])
		img = img.flatten()
		img[img>cutoff] = cutoff
		img = img.reshape([w,h])

		#assert np.max(img) > 0.0 
		img = img/cutoff
		#img_visual = 255*(self.depth_image_raw/cutoff)
		img = np.array(img, dtype=np.float32)
		
		#img_visual = np.array(img_visual, dtype=np.uint8)
		#img_visual = cv2.equalizeHist(img_visual)
		#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_cutoff.png', img_visual)
        #for i in range(3):
        #    img[:,:,i] = cv2.equalizeHist(img[:,:,i])
		return img 

	def process_odom(self, odom_msg, reset_step):
		if(reset_step):
			self.previous_pos_x = odom_msg.pose.pose.position.x
			self.previous_pos_y = odom_msg.pose.pose.position.y
			self.total_distance = 0

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
		#print('Goal distance:', self.goal_distance)

		d_increment = math.sqrt((pos_x - self.previous_pos_x)**2 + (pos_y - self.previous_pos_y)**2)
		self.total_distance = self.total_distance + d_increment
		#print("Total distance traveled is ", self.total_distance)


		self.previous_pose_x = pos_x
		self.previous__pose_y = pos_y

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
	pic4rl_environment = Pic4rlEnvironment()
	pic4rl_environment.spin()

	pic4rl_environment.get_logger().info('Node spinning ...')
	rclpy.spin_once(pic4rl_environment)

	pic4rl_environment.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
