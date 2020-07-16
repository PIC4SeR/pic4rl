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

from pic4rl_sensors.Sensor import OdomSensor, LaserScanSensor

import numpy as np

import time 
import collections

class Pic4rlGymGazEnv(Node):
	def __init__(self,odom = False,lidar = False):

		super().__init__('Pic4rlGymGazEnv')

		self.state_history = collections.deque(maxlen=2)
		self.observation_history = collections.deque(maxlen=2)
		self.state = {}
		self.observation  = {}

		#rclpy.logging.set_logger_level('Pic4rlGymGazEnv', 10)
		self.__init__gym()

		self.__init__gazebo()

		self.odom = odom
		self.lidar = lidar

		self.__init__sensors()

	"""################
	# gym related
	################"""

	def __init__gym(self):
		#self.action_space = spaces.Box()
		#self.observation_space = spaces.Box()
		pass

	def step(self,action):
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

		self.update_state() # Take new data from sensor, clean them 

		observation, done = self.get_observation()
		reward = self.get_reward()
		info = None

		return observation, reward, done, info

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
		self.get_logger().debug("Reset request received ...")
		self.get_logger().debug("Resetting world ...")
		self.reset_state()
		self.reset_world()
		self.get_goal()
		data_retrieved = False
		while not data_retrieved:
			self.unpause()
			self.spin_with_timeout()
			self.pause()	
			try:
				for sensor in self.sensors:
					sensor.data
				data_retrieved = True
			except:
				self.get_logger().debug("Waiting for data...")
				pass
		self.update_state()
		observation = self.get_only_observation()
		self.observation_history.append(self.observation.copy())
		return observation
		#self.respawn_entity(request.goal_pos_x, request.goal_pos_y)

	def get_reward(self):

		raise NotImplementedError
	
	def get_observation(self):

		raise NotImplementedError

	def get_only_observation(self):

		raise NotImplementedError

	def get_goal(self):

		raise NotImplementedError

	def render(self):
		pass
		#raise NotImplementedError

	def define_action_space(self):
		"""
		Here should be defined all actions:
		e.g. angular eand linear velocites with related bounding box
		"""

		raise NotImplementedError

	def define_state_space(self):
		"""
		Here should be defined all variables componing the state
		(Also previous step state should be included)
		Both internal and external (e.g. sensors)
		"""

		raise NotImplementedError

	def update_state(self):
		for sensor in self.sensors:
			processes_data = sensor.process_data()
			try:
				for key in processes_data.keys(): #Remove previous data
					self.state.pop(key) #I don't use .clean because 
										# I don't wanna lose goal related info
			except KeyError:
				pass #first time an error rise
			self.state.update(processes_data)

		self.state_history.append(self.state.copy())



	def uupdate_state(self):
		print("ççççççççççççççççççççççççççççççççççò")
		try:
			print("PRIMA " +str(self.state['odom_pos_x']))
		except:
			pass
		for sensor in self.sensors:
			processes_data = sensor.process_data()
			print("0_______")
			print(processes_data.keys())
			
			print("1_______")
			print(self.state.keys())
			try:
				for key in processes_data.keys(): #Remove previous data
					self.state.pop(key) #I don't use .clean because 
										# I don't wanna lose goal related info
			except KeyError:
				pass #first time an error rise
			print("2_______")
			print(self.state.keys())
			self.state.update(processes_data)
			print("3_______")
			print(self.state.keys())
		self.state_history.append(self.state)

		try:
			print(self.state_history[0]["odom_pos_x"])
			print(self.state_history[1]["odom_pos_x"])
		except:
			pass
		print("DOPO " +str(self.state['odom_pos_x']))

	def reset_state(self):
		for sensor in self.sensors:
			del sensor.data

		
	"""################
	# Ros 
	################"""

	def spin_with_timeout(self,	timeout_sec = 0.1):
		"""This function provides a way to spin only for a certain
		amount of time"""
		start = time.time()
		self.get_logger().debug("Spinning")
		rclpy.spin_until_future_complete(self,rclpy.Future(),timeout_sec=timeout_sec)
		debug_string = "Spinned for " + str(time.time()-start)
		self.get_logger().debug(debug_string)

	"""################
	# Rl related
	################"""

	"""################
	# Sensors
	################"""

	def __init__sensors(self):
		self.sensors = []

		if self.odom:
			odom_sensor = OdomSensor()
			self.create_subscription(*odom_sensor.add_subscription())
			self.sensors.append(odom_sensor)

		if self.lidar:
			lidar_sensor = LaserScanSensor()
			self.create_subscription(*lidar_sensor.add_subscription())
			self.sensors.append(lidar_sensor)

		print("Following sensors are used:")
		for sensor in self.sensors:
			print(sensor.name)

	"""################
	# Gazebo services
	################"""

	def __init__gazebo(self):

		# Service clients
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
			self.get_logger().info('service not available, waiting again...')
		self.unpause_physics_client.call_async(req) 

	def delete_entity(self, entity_name):
		req = DeleteEntity.Request()
		#req.name = self.entity_name
		req.name = entity_name
		while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('service not available, waiting again...')

		self.delete_entity_client.call_async(req)
		self.get_logger().debug('Entity deleting request sent ...')

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
