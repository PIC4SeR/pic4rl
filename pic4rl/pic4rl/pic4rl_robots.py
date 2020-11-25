#!/usr/bin/env python3

import time
import rclpy
from rclpy.executors import MultiThreadedExecutor

# TO DO timeout_sec to be added as argument
class SpinWithTimeout():
	def __init__(self, parent_node):
		self.parent_node = parent_node
		parent_node.spin_with_timeout = self.spin_with_timeout

	# TO DO timeout_sec to be added as argument
	def spin_with_timeout(self,	timeout_sec = 0.1):
		"""This function provides a way to spin only for a certain
		amount of time"""
		start = time.time()
		self.parent_node.get_logger().debug("Spinning")
		rclpy.spin_until_future_complete(self.parent_node,
										rclpy.Future(),
										MultiThreadedExecutor(num_threads=4),
										timeout_sec=timeout_sec)
		debug_string = "Spinned for " + str(time.time()-start)
		self.parent_node.get_logger().debug(debug_string)



from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist

class Differential2Twist():
	def __init__ (self, parent_node):
		self.parent_node = parent_node
		self.parent_node.get_logger().debug("Differential2Twist.")
		parent_node.send_cmd_command = self.send_cmd_command
		qos = QoSProfile(depth=10)

		self.parent_node.cmd_vel_pub = self.parent_node.create_publisher(
			Twist,
			'cmd_vel',
			qos)

	def send_cmd_command(self,linear_speed, angular_speed):
		twist = Twist() #void instance created
		
		if (linear_speed or linear_speed) is None:
			pass #null action (0,0)
		else:
			twist.linear.x = linear_speed
			twist.angular.z = angular_speed
		self.parent_node.cmd_vel_pub.publish(twist)

