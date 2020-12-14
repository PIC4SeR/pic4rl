#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from pic4rl.pic4rl_3_robot import Pic4rlRobot
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import time 
import collections

def main(args=None):
	rclpy.init()

	executor = MultiThreadedExecutor(num_threads=4)
	print(executor)

	env = Pic4rlRobot(executor)
	while True:
		env.step([0,0])
	#pic4rl_test.main()

	"""pic4rl_test.get_logger().info('Node spinning ...')
	rclpy.spin_once(pic4rl_test)

	pic4rl_test.destroy()
	rclpy.shutdown()"""
	env.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
