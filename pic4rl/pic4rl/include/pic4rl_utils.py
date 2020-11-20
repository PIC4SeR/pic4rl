#!/usr/bin/env python3

import time
import rclpy
from rclpy.executors import MultiThreadedExecutor

# TO DO timeout_sec to be added as argument
def spin_with_timeout(parent_self,	timeout_sec = 0.1):
	"""This function provides a way to spin only for a certain
	amount of time"""
	start = time.time()
	parent_self.get_logger().debug("Spinning")
	rclpy.spin_until_future_complete(parent_self,
									rclpy.Future(),
									MultiThreadedExecutor(num_threads=4),
									timeout_sec=timeout_sec)
	debug_string = "Spinned for " + str(time.time()-start)
	parent_self.get_logger().debug(debug_string)

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