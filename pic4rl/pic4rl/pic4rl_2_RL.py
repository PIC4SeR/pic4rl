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

import pic4rl.pic4rl_utils
from pic4rl.pic4rl_utils import SpinWithTimeout
from pic4rl.pic4rl_utils import Differential2Twist

import pic4rl.pic4rl_services
from pic4rl.pic4rl_services import ResetWorldService, PauseService , UnpauseService
		
import pic4rl.pic4rl_sensors
from pic4rl.pic4rl_sensors import OdomSensor, pose_2_xyyaw
from pic4rl.pic4rl_sensors import CmdVelInfo
from pic4rl.pic4rl_sensors import LaserScanSensor, clean_laserscan, laserscan_2_list

from pic4rl.pic4rl_sensors import s7b3State
class Pic4rl(Node):
	def __init__(self):
		super().__init__("pic4rl")
		rclpy.logging.set_logger_level('pic4rl', 10)

		self.robot = s7b3State(self)
		self.initialization()

	"""###########
	# TOP LEVEL
	###########"""

	def initialization(self,args=None):
		self.get_logger().debug('Initialization ...')
		self.initialize_ros()
		self.initialize_gazebo_services()
		self.initialize_sensors()

	def reset(self,args=None):

		self.reset_gazebo()
		self.collect_data_by_spinning(0.3)

	def step(self,action):

		self.send_action_to_Gazebo(action)
		self.collect_data_by_spinning()
		self.raw_data_to_state()

	"""#
	# -1
	#"""

	# INITIALIZATION

	def initialize_ros(self,args=None):
		# Add spin_with_timeout function
		SpinWithTimeout(self)

	def initialize_gazebo_services(self,args=None):

		ResetWorldService(self)
		PauseService(self)
		self.pause() # So that the simulation start paused
		UnpauseService(self)

		Differential2Twist(self)

	# rather robot
	def initialize_sensors(self,args=None):

		self.robot_state.initialize_sensors()
		#self.odom_sensor = OdomSensor(self)
		#self.laser_scan_sensor = LaserScanSensor(self)
		#self.cmd_vel_sensor = CmdVelInfo(self) #only for test purposes

	# RESET

	# Reset Gazebo
	def reset_gazebo(self,args=None):

		self.reset_world()
		# reset goal
		# reset other elements if any

	# Collect data by node spinning
	def collect_data_by_spinning(self, timeout_sec = 0.1):
		self.unpause()
		self.spin_with_timeout(timeout_sec)
		self.pause()

	# Get new state from gazebo 
	def raw_data_to_state(self,args=None):

		self.robot.get_state()

	# Process state and obtain observation
	def get_observation(self,args=None):
		
		self.robot.get_observation()

	# STEP

	# Convert action to Twist(or other) msg and send to gazebo
	def send_action_to_Gazebo(self,action):

		self.send_cmd_command(action[0],action[1])


	# Collect data by node spinning
	# See in RESET

	# Process state and obtain observation
	# See in RESET

	# Compute reward from state (history)
	def get_reward(self,args=None):
		pass


def main(args=None):
	rclpy.init()
	pic4rl = Pic4rl()
	#	rclpy.spin()

	pic4rl.get_logger().info('Node spinning once...')
	#rclpy.spin_once(pic4rl)
	try:
		for i in range(3):
			pic4rl.reset()
			pic4rl.step([1.0,1.0])
			#print(type(pic4rl.odom_sensor.data))
			#print(type(pic4rl.cmd_vel_sensor.data))
			time.sleep(5)
			#pic4rl.spin_with_timeout()
			#pic4rl.send_cmd_command(1.0,1.0)
	finally:
		pic4rl.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
