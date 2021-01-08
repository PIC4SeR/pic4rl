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

class SpawnEntityService():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Spawn Entity client created.")
		parent_node.spawn_entity_client = parent_node.create_client(SpawnEntity, 'spawn_entity')
		parent_node.spawn_entity = self.spawn_entity

	def spawn_entity(self, pose = None, name = None, entity_path = None, entity = None):
		if not pose:
			pose = Pose()

		req = SpawnEntity.Request()
		req.name = name
		if entity_path:
			entity = open(entity_path, 'r').read()
		req.xml = entity
		req.initial_pose = pose
		while not self.parent_node.spawn_entity_client.wait_for_service(timeout_sec=1.0):
			self.parent_node.get_logger().info('\'spawn_entity\' service not available, waiting again...')
		self.parent_node.spawn_entity_client.call_async(req)

class DeleteEntityService():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Unpause client created.")
		parent_node.delete_entity_client = parent_node.create_client(DeleteEntity, 'delete_entity')
		parent_node.delete_entity = self.delete_entity

	def delete_entity(self, name = None):
		req = DeleteEntity.Request()
		req.name = name
		while not self.parent_node.delete_entity_client.wait_for_service(timeout_sec=1.0):
			self.parent_node.get_logger().info('\'delete_entity\' service not available, waiting again...')

		self.parent_node.delete_entity_client.call_async(req)
		self.parent_node.get_logger().debug('Entity deleting request sent ...')
