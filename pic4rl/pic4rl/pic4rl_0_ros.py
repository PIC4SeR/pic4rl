#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time 
from rclpy.executors import MultiThreadedExecutor

class Pic4rlROS(Node):
	def __init__(self,
				executor,
				node_name = "Pic4rlNode"):
	# TO DO add debug level as parameter
		super().__init__(node_name)
		rclpy.logging.set_logger_level(node_name, 10)
		self.get_logger().info('Class: Pic4rlROS')


		self.executor = MultiThreadedExecutor(num_threads=4)
		self.executor = executor
		self.other_nodes_added = False




	# TO DO timeout_sec to be added as argument
	def spin_with_timeout(self,	timeout_sec = 0.1):
		if not self.other_nodes_added:
			self.executor.add_node(self)
			for sensor in self.sensors:
				self.executor.add_node(sensor)
			#self.executor.add_node(self.sensor)

		"""This function provides a way to spin only for a certain
		amount of time"""
		start = time.time()
		self.get_logger().debug("Spinning")
		self.executor.spin_until_future_complete(rclpy.Future(),timeout_sec=timeout_sec)
		rclpy.spin_until_future_complete(self,rclpy.Future(),timeout_sec=timeout_sec)
		debug_string = "Spinned for " + str(time.time()-start)
		self.get_logger().debug(debug_string)



def main(args=None):
	rclpy.init()
	pic4rl_ros = Pic4rlROS()
#	rclpy.spin()

	pic4rl_ros.get_logger().info('Node spinning once...')
	rclpy.spin_once(pic4rl_ros)

	pic4rl_ros.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
