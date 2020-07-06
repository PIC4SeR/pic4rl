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

from pic4rl.pic4rl_gymgazebo import Pic4rlGymGazEnv
from pic4rl_sensors.Sensor import OdomSensor, LaserScanSensor

class Pic4rlTurtleBot3(Pic4rlGymGazEnv):
	def __init__(self):
		super().__init__(
			odom = True,
			lidar = True)

	"""################
	# gym related
	################"""


	def step():

		raise NotImplementedError

	def reset():

		raise NotImplementedError

	def get_reward():

		raise NotImplementedError

	def render():
		pass
		#raise NotImplementedError

	"""################
	# Ros 
	################"""

	"""################
	# Rl related
	################"""
	
	def get_goal(self):
		x = random.uniform(-3,3)
		y = random.uniform(-3,3)
		self.get_logger().info("New goal")
		#self.get_logger().info("New goal: (x,y) : " + str(x) + "," +str(y))
		self.goal_pose_x = x
		self.goal_pose_y = y
		return x,y

	"""################
	# Sensors
	################"""



	"""################
	# Gazebo services
	################"""


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
