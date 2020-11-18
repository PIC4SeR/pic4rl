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
from pic4rl.pic4rl_2_RL import Pic4rlGym
import numpy as np
import time 
import collections
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class Pic4rlRobot(Pic4rlGym):
	def __init__(self,
				executor):
		super().__init__(
				executor)
		self.get_logger().info('Class: Pic4rlRobot')
		self.add_sensors()

	##############################################
	# Actuation
	##############################################

	def _step(self,action):
		twist = Twist()
		twist.linear.x = 0.1
		self.cmd_vel_pub.publish(twist)


	def add_sensors(self):
		#self.sensor = Listener()
		pass

"""def main(args=None):
	rclpy.init()
	pic4rl_sensors = Pic4rlSensors()

	pic4rl_sensors.get_logger().info('Node spinning ...')
	rclpy.spin_once(pic4rl_sensors)

	pic4rl_sensors.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()"""
