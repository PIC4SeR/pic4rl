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
import numpy as np
import time 
import collections
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class ResetWorldService():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Reset world client created.")
		parent_node.reset_world_client = parent_node.create_client(Empty, 'reset_world')
		parent_node.reset_world = self.reset_world

	def reset_world(self):
		req = Empty.Request()
		while not self.parent_node.reset_world_client.wait_for_service(timeout_sec=1.0):
			self.parent_node.get_logger().info('\'reset_world\'service not available, waiting again...')
		self.parent_node.reset_world_client.call_async(req)    

class PauseService():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Pause client created.")
		parent_node.pause_physics_client = parent_node.create_client(Empty, 'pause_physics')
		parent_node.pause = self.pause

	def pause(self):
		req = Empty.Request()
		while not self.parent_node.pause_physics_client.wait_for_service(timeout_sec=1.0):
			self.parent_node.get_logger().info('\'pause_physics\' service not available, waiting again...')

		self.parent_node.pause_physics_client.call_async(req) 

class UnpauseService():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Unpause client created.")
		parent_node.unpause_physics_client = parent_node.create_client(Empty, 'unpause_physics')
		parent_node.unpause = self.unpause

	def unpause(self):
		req = Empty.Request()
		while not self.parent_node.unpause_physics_client.wait_for_service(timeout_sec=1.0):
			self.parent_node.get_logger().info('\'unpause_physics\' service not available, waiting again...')

		self.parent_node.unpause_physics_client.call_async(req) 


