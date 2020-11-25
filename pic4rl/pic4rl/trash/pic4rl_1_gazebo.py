#!/usr/bin/env python3

import rclpy
from rclpy.qos import QoSProfile
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from pic4rl.pic4rl_0_ros import Pic4rlROS
from geometry_msgs.msg import Twist
import numpy as np
import time 
import collections

class Pic4rlGazebo(Pic4rlROS):
	def __init__(self,
				executor):
		super().__init__(
				executor)
		self.start_gazebo_services()
		self.get_logger().info('Class: Pic4rlGazebo')

		"""################
		# Gazebo topics
		################"""

		qos = QoSProfile(depth=10)
		self.cmd_vel_pub = self.create_publisher(
			Twist,
			'cmd_vel',
			qos)

	"""################
	# Gazebo services
	################"""

	def start_gazebo_services(self):

		# Service clients
		self.get_logger().info('Starting gazebo services...')
		self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
		self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
		self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
		self.reset_world_client = self.create_client(Empty, 'reset_world')
		self.pause_physics_client = self.create_client(Empty, 'pause_physics')
		self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

	def pause(self):
		req = Empty.Request()
		while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		
		self.pause_physics_client.call_async(req) 

	def unpause(self):
		req = Empty.Request()
		while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('\'unpause_physics\' service not available, waiting again...')
		self.unpause_physics_client.call_async(req) 

	def delete_entity(self, entity_name):
		req = DeleteEntity.Request()
		#req.name = self.entity_name
		req.name = entity_name
		while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')

		self.delete_entity_client.call_async(req)
		self.get_logger().debug('Entity deleting request sent ...')

	# TO DO adjust the function
	def spawn_entity(self,pose = None, name = None, entity_path = None, entity = None):
		if not pose:
			pose = Pose()
		req = SpawnEntity.Request()
		req.name = name
		if entity_path:
			entity = open(entity_path, 'r').read()
		req.xml = entity
		req.initial_pose = pose
		while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		self.spawn_entity_client.call_async(req)

	def reset_world(self):
		req = Empty.Request()
		while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')
		self.reset_world_client.call_async(req)    
		#time.sleep(1)


"""def main(args=None):
	rclpy.init()
	pic4rl_gazebo = Pic4rlGazebo()

	pic4rl_gazebo.get_logger().info('Node spinning ...')
	rclpy.spin_once(pic4rl_gazebo)

	pic4rl_gazebo.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()"""
