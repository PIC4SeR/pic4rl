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
from rclpy.node import Node

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from pic4rl.pic4rl_1_gazebo import Pic4rlGazebo

import numpy as np
import time 
import collections

class Pic4rlGym(Pic4rlGazebo):
	def __init__(self,
				executor):
		super().__init__(
				executor)
		self.get_logger().info('Class: Pic4rlGym')
		self.local_step = 0
		self.global_step = 0
		self.episode = 0

	def step(self,action):
			self.global_step += 1
			self.local_step += 1

			"""This method should provide the command to be sent to gazebo
			and handled interanlly via gazebo_step method
			"""
			self.get_logger().debug("unpausing...")
			self.unpause()
			self.get_logger().debug("publishing twist...")

			self._step(action)

			self.spin_with_timeout()
			self.get_logger().debug("pausing...")
			self.pause()	

			#self.update_state() # Take new data from sensor, clean them 

			#observation, done, done_info = self.get_observation()
			#reward = self.get_reward(done, done_info)
			#info = None

			#return observation, reward, done, info

	# NotImplemented
	def _step(self, action):
		"""
		This method must be implemented in the specific 
		environment
		e.g. for mobile robots it should publish a 
		twist message over /cmd_vel topic, hence 
		mapping the action from the rl agent to 
		the twist message
		"""

		raise NotImplementedError

	def reset(self):
		self.episode += 1

		self.get_logger().debug("Reset request received ...")
		self.get_logger().debug("Resetting world ...")
		self.get_logger().info("Reset...")
		self.reset_state()
		self.reset_world()
		self.get_goal()
		self.unpause()
		self.spin_with_timeout()
		self.pause()
		#data_retrieved = False
		#while not data_retrieved:
		#	self.unpause()
		#	self.spin_with_timeout()
		#	self.pause()	
		#	try:
		#		for sensor in self.sensors:
		#			sensor.data
		#		data_retrieved = True
		#	except:
		#		self.get_logger().debug("Waiting for data...")
		#		pass
		#self.update_state()
		#observation = self.get_only_observation()
		#self.observation_history.append(self.observation.copy())
		#return observation


	# NotImplemented
	def render(self):
		pass
		#raise NotImplementedError



	"""def main(args=None):
		rclpy.init()
		pic4rl_sensors = Pic4rlSensors()

		pic4rl_sensors.get_logger().info('Node spinning ...')
		rclpy.spin_once(pic4rl_sensors)

		pic4rl_sensors.destroy()
		rclpy.shutdown()

	if __name__ == '__main__':
		main()"""
