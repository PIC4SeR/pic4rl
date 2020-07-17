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
from pic4rl_sensors.Sensor import OdomSensor, LaserScanSensor

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
			lidar = True)
		self.__init__spaces()

		qos = QoSProfile(depth=10)

		self.cmd_vel_pub = self.create_publisher(
			Twist,
			'cmd_vel',
			qos)



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

		# Distance
		distance_low = [0]
		distance_high = [10]

		# Angle
		angle_low = [-math.pi]
		angle_high = [math.pi]

		self.observation_space = spaces.Box(
			low = np.array(lidar_low + 
							distance_low +
							angle_low,
							dtype=np.float32),
			high = np.array(lidar_high + 
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
			twist.linear.x = float(action[0])
			twist.angular.z = float(action[1])
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

		return np.array(observation)

	def get_observation(self):

		observation = self.get_only_observation()
		self.observation_history.append(self.observation.copy())
		done = self._check_done()

		return observation, done

	def _check_done(self):

		done = False

		if min(self.state["scan_ranges"]) <= 0.13:
			"""
			self.state -> has all 359 lidar ranges
			self.observation -> has reduced number of ldiar ranges
			"""
			done = True
			self.get_logger().info("Collision")
		
		if self.observation["goal_distance"] < 0.1:
			done = True
			self.get_logger().info("Goal")

		return done

	def get_reward(self):
		"""
		The reward value has to be computed, using state/observation
		observation_history is a 2 element deque object containing 
		current and previous observation dictionary.
		Same holds for state_history
		"""

		distance_delta = self.observation_history[0]["goal_distance"] - self.observation_history[1]["goal_distance"] 

		return distance_delta

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

	def render(self):

		pass

	"""################
	# Ros 
	################"""

	"""################
	# Rl related
	################"""
	
	def get_goal(self):
		x = random.uniform(-3,3)
		y = random.uniform(-3,3)
		info_string = "New goal!! " + str([x, y])
		self.get_logger().info(info_string)
		#self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
		self.state.update({"goal_pos_x":x,
							"goal_pos_y":y})

	"""################
	# Sensors
	################"""



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
