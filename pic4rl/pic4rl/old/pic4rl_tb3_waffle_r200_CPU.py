#!/usr/bin/env python3

"""
This class is to be inherited by all the pic4rl enviornments  
	Ros
	Gym
	Rl related
	Sensors
	Gazebo 
"""

import rclpy
from rclpy.qos import QoSProfile

from pic4rl.pic4rl_gymgazebo import Pic4rlGymGazEnv
from pic4rl_sensors.Sensor import OdomSensor, LaserScanSensor, RealSenseSensor

from geometry_msgs.msg import Twist



import math
import numpy as np 
import gym
from gym import spaces
import random
import time

class Pic4rlTurtleBot3(Pic4rlGymGazEnv):
	def __init__(self):
		super().__init__(
			odom = True,
			lidar = True,
			realsense = True)
		self.__init__spaces()

		qos = QoSProfile(depth=10)

		self.cmd_vel_pub = self.create_publisher(
			Twist,
			'cmd_vel',
			qos)

		self.__init__sensors()


	"""################
	# gym related
	################""" 
	def __init__spaces(self):
		#Action space

		#Linear velocity
		lin_speed_low = [+0] 
		lin_speed_high = [+0.2]

		#Angular velocity
		ang_speed_low = [-1.7]
		ang_speed_high = [1.7]

		self.action_space = spaces.Box(
			low = np.array(lin_speed_low + 
							ang_speed_low,
							dtype=np.float32),
			high = np.array(lin_speed_high +
							ang_speed_high,
							dtype=np.float32),
			#shape=(1,),
			dtype = np.float32
			)

		#Observation space

		#Lidar points
		n_points = 60
		lidar_low = [0] * n_points
		lidar_high = [3.5] * n_points


		#Camera points
		n_points = 60
		camera_low = [0] * n_points
		camera_high = [3.5] * n_points

		# Distance
		distance_low = [0]
		distance_high = [5]

		# Angle
		angle_low = [-math.pi]
		angle_high = [math.pi]

		self.observation_space = spaces.Box(
			low = np.array(lidar_low + 
							camera_low +
							distance_low +
							angle_low,
							dtype=np.float32),
			high = np.array(lidar_high + 
							camera_high +
							distance_high +
							angle_high,
							dtype=np.float32),
			dtype = np.float32
			)

	def _step(self,action):
		twist = Twist() #void instance created
		
		if action is None:
			pass #null action
		else:
			#linear_speed = float(action[0][0][0])/255 * 0.2
			#angular_speed = float(action[0][0][1])/255 * 3.4 - 1.7

			# /2 +1 is added beacuse the ouput of the actor uses tanh
			# hence the action for linear x is [-0.2, 0.2] and we 
			# want to transform it to [0, 0.2]
			twist.linear.x = float(action[0])/2 + 0.1 
			twist.angular.z = float(action[1])

			#twist.linear.x = linear_speed
			#twist.angular.z = angular_speed
		self.cmd_vel_pub.publish(twist)

	def get_only_observation(self):
		"""
		self.state may have these keys
		odom_pos_x 	(float)
		odom_pos_y 	(float)
		odom_yaw	(float)
		scan_ranges (list of float)
		goal_pos_x 	(float)
		goal_pos_y 	(float)
		"""
		observation = []

		# Adding angle and distance
		observation += self.process_odom() 

		# Reducing lidar ranges number
		observation += self.process_laserscan()

		# Selecting some points from depth camera r200
		observation += self.process_depth_image()

		"""# Adding angle and distance
		goal_distance, goal_angle = self.process_odom() 
		depth_image = self.state["depth_image"]
		tf_goal_distance = tf.math.multiply(tf.ones(self.image_size, dtype = tf.float32), self.goal_distance)
		tf_goal_angle = tf.math.multiply(tf.ones(self.image_size, dtype = tf.float32), self.goal_angle)
		#print(tf_goal_distance)
		state_tf = tf.stack([tf_goal_distance,tf_goal_angle,depth_image], axis=2)
		tf.print("________________________________________-")
		print(state_tf.shape)
		# Reducing lidar ranges number
		#observation += self.process_laserscan()

		return state_tf"""
		return np.array(observation)

	def get_observation(self):

		observation = self.get_only_observation()
		self.observation_history.append(self.observation.copy())
		done, done_info = self._check_done()

		return observation, done, done_info

	def _check_done(self):

		done = False
		done_info = 0

		if min(self.state["scan_ranges"]) <= 0.19:
			"""
			self.state -> has all 359 lidar ranges
			self.observation -> has reduced number of ldiar ranges
			"""
			done = True
			done_info = 2
			self.get_logger().info("Collision")
		
		if self.observation["goal_distance"] < 0.1:
			done = True
			done_info = 1
			self.get_logger().info("Goal")

		"""done_info
		1 --> goal
		2 --> 2 collision
		"""

		return done, done_info

	def get_reward(self, done, done_info):
		"""
		The reward value has to be computed, using state/observation
		observation_history is a 2 element deque object containing 
		current and previous observation dictionary.
		Same holds for state_history
		"""
		if not done:
			distance_delta = self.observation_history[0]["goal_distance"] - self.observation_history[1]["goal_distance"] 	
			#The reward is computed to give higher negatives values when distance increase. Secondo term is 
			# put to make the higher reward equal to 0
			distance_reward = -(0.01/(distance_delta+0.05)) - (0.01/(0.04+0.05)) 
			return distance_delta
		else:
			if done_info == 1:
				# GOAL
				return 5
			elif done_info == 2:
				# Collision
				return -5
			else:
				raise ValueError("done_info is out of range")


	def process_odom(self):
		goal_pos_x = self.state["goal_pos_x"]
		goal_pos_y = self.state["goal_pos_y"]
		pos_x = self.state["odom_pos_x"]
		pos_y = self.state["odom_pos_y"]
		yaw = self.state["odom_yaw"]

		goal_distance = math.sqrt(
			(goal_pos_x-pos_x)**2
			+ (goal_pos_y-pos_y)**2)

		path_theta = math.atan2(
			goal_pos_y-pos_y,
			goal_pos_x-pos_x)

		goal_angle = path_theta - yaw

		if goal_angle > math.pi:
			goal_angle -= 2 * math.pi

		elif goal_angle < -math.pi:
			goal_angle += 2 * math.pi

		self.observation["goal_distance"] = goal_distance
		self.observation["goal_angle"] = goal_angle

		return goal_distance, goal_angle

	def process_laserscan(self, n_points = 60):
		scan_range = self.state["scan_ranges"]
		modified_scan_range = []

		for index in range(n_points):
			modified_scan_range.append(scan_range[int(index*len(scan_range)/n_points)])
		
		self.observation["scan_ranges"] = modified_scan_range
		return modified_scan_range

	def process_depth_image(self, n_points = 60):
		depth_image = self.state["depth_image"]
		#print(type(depth_image))
		#print(depth_image.shape)
		#print(depth_image)
		#print("**********************************+")
		#print(depth_image.flatten())
		#reduced_depth_image = 
		#raise ValueError("testtesttesttest____________")
		depth_image_flatten = depth_image.flatten()
		reduced_depth_image = []
		space_between_points = round(depth_image_flatten.size/n_points)
		for i in range(len(depth_image_flatten)):
			if i%space_between_points == 0:
				reduced_depth_image.append(depth_image_flatten[i])

		#reduced_depth_image= np.array(reduced_depth_image)
		return reduced_depth_image





	def render(self):

		pass

	"""################
	# Ros 
	################"""

	"""################
	# Rl related
	################"""
	
	def get_goal(self):
		x = random.uniform(-2,2)
		y = random.uniform(-2,2)
		info_string = "New goal!! " + str([x, y])
		self.get_logger().info(info_string)
		#self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
		self.state.update({"goal_pos_x":x,
							"goal_pos_y":y})

	"""################
	# Sensors
	################"""

	def __init__sensors(self):
		self.sensors = []

		if self.odom:
			self.odom_sensor = OdomSensor()
			self.create_subscription(*self.odom_sensor.add_subscription())
			self.sensors.append(self.odom_sensor)

		if self.lidar:
			self.lidar_sensor = LaserScanSensor()
			self.create_subscription(*self.lidar_sensor.add_subscription())
			self.sensors.append(self.lidar_sensor)

		if self.realsense:
			self.realsense_sensor = RealSenseSensor()
			self.create_subscription(*self.realsense_sensor.add_subscription())
			self.sensors.append(self.realsense_sensor)

		print("Following sensors are used:")
		for sensor in self.sensors:
			print(sensor.name)


	"""################
	# Gazebo services
	################"""


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
