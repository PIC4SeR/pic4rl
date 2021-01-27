#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import random 
import math 
import os 
from ament_index_python.packages import get_package_share_directory

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

from pic4rl.sensors.pic4rl_sensors import pose_2_xyyaw
from pic4rl.sensors.pic4rl_sensors import clean_laserscan, laserscan_2_list, laserscan_2_n_points_list
from pic4rl.sensors.pic4rl_sensors_class import Sensors

from pic4rl.tasks.pic4rl_states import OdomLidarState, OdomState, OdomDepthState
from pic4rl.tasks.pic4rl_end_of_episodes import OdomGoalLidarCollision
from pic4rl.tasks.pic4rl_observations import  OdomLidarObs, OdomObs, OdomDepthObs
from pic4rl.tasks.pic4rl_goals import RandomGoal
from pic4rl.tasks.pic4rl_locomotions import WheeledRobot
from pic4rl.tasks.pic4rl_rewards import SimpleDistanceReward

import gym
from gym import spaces


import collections
import yaml


yaml_path = os.path.join(get_package_share_directory('pic4rl'),'config', "task_param.yaml")

with open(yaml_path) as file:
	params = yaml.load(file)

arg_end_of_episode = params["pic4rl"]["task"]["end_of_episode"]
arg_goal = params["pic4rl"]["task"]["goal"]
arg_reward = params["pic4rl"]["task"]["reward"]
arg_observation = params["pic4rl"]["task"]["observation"]
arg_state = params["pic4rl"]["task"]["state"]
arg_locomotion = params["pic4rl"]["task"]["locomotion"]

end_of_episode = None
end_of_episode = OdomGoalLidarCollision if (arg_end_of_episode == "OdomGoalLidarCollision") else end_of_episode

goal = None
goal = RandomGoal if (arg_goal == "RandomGoal") else goal

reward = None
reward = SimpleDistanceReward if (arg_reward == "SimpleDistanceReward") else reward

observation = None
observation = OdomLidarObs if (arg_observation == "OdomLidarObs") else observation
observation = OdomObs if (arg_observation == "OdomObs") else observation
observation = OdomDepthObs if (arg_observation == "OdomDepthObs") else observation

state = None
print(arg_state)
state = OdomLidarState if (arg_state == "OdomLidarState") else state
state = OdomState if (arg_state == "OdomState") else state
state = OdomDepthState if (arg_state == "OdomDepthState") else state

locomotion = None
locomotion = WheeledRobot if (arg_locomotion == "WheeledRobot") else locomotion


class Pic4rlTask(
	end_of_episode,
	goal, 
	reward,
	observation,
	state,
	locomotion,
	Sensors
	):

	# This class refers to a navigation task in which
	# a ground mobile robot uses
	#	Odometry 	(<-- /odom)
	#	Lidar 		(<-- /scan)
	# 
	# To reach goals that are spawned.
	# The robot can have differential/holonomic locomotion
	# system
	#
	#
	#
	def __init__(self):
		self.get_logger().debug('[LidarNavigation] Initialization.')
		Sensors.__init__(self, 
				generic_laser_scan_sensor = True,
				odometry_sensor = True)
		locomotion.__init__(self) #currently only differential
		state.__init__(self)
		observation.__init__(self)
		reward.__init__(self)
		goal.__init__(self)
		end_of_episode.__init__(self)
