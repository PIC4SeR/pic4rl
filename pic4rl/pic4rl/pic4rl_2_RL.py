#!/usr/bin/env python3

"""
This class is to be inherited by all the pic4rl enviornments  
"""

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty

import numpy as np
import time 
import collections

from rclpy.executors import MultiThreadedExecutor

import pic4rl.include.pic4rl_utils
from pic4rl.include.pic4rl_utils import SpinWithTimeout


class Pic4rl(Node):
	def __init__(self):
		super().__init__("pic4rl")
		rclpy.logging.set_logger_level('pic4rl', 10)
		self.initialization()

	"""###########
	# TOP LEVEL
	###########"""

	def initialization(self,args=None):
		self.get_logger().debug('Initialization ...')
		self.initialize_ros()

	def step(self,args=None):
		pass

	def reset(self,args=None):
		pass


	"""#
	# -1
	#"""

	# INITIALIZATION

	def initialize_ros(self,args=None):
		SpinWithTimeout(self)

	def initialize_gazebo_services(self,args=None):
		pass

	def initialize_sensors(self,args=None):
		pass


	# RESET

	# Reset Gazebo
	def reset_gazebo(self,args=None):
		pass

	# Collect data by node spinning
	def collect_data_by_spinning(self,args=None):
		pass

	# Get new state from gazebo 
	def raw_data_to_state(self,args=None):
		pass

	# Process state and obtain observation
	def get_observation(self,args=None):
		pass

	# STEP

	# Convert action to Twist(or other) msg and send to gazebo
	def send_action_to_Gazebo(self,args=None):
		pass

	# Collect data by node spinning
	# See in RESET

	# Process state and obtain observation
	# See in RESET

	# Compute reward from state (history)
	def get_reward(self,args=None):
		pass

	def __function__(self,args=None):
		pass

def main(args=None):
	rclpy.init()
	executor = MultiThreadedExecutor(num_threads=4)
	pic4rl = Pic4rl()
#	rclpy.spin()

	pic4rl.get_logger().info('Node spinning once...')
	#rclpy.spin_once(pic4rl)
	pic4rl.spin_with_timeout()
	pic4rl.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
