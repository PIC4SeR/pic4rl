#!/usr/bin/env python3

"""
This class is to be inherited by all the pic4rl enviornments  
"""

import rclpy
from rclpy.node import Node
import random

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty

import numpy as np
import time 
import math

from rclpy.executors import MultiThreadedExecutor

import pic4rl.pic4rl_utils
from pic4rl.pic4rl_utils import SpinWithTimeout
from pic4rl.pic4rl_utils import Differential2Twist

import pic4rl.pic4rl_services
from pic4rl.pic4rl_services import ResetWorldService, PauseService , UnpauseService
		
import pic4rl.pic4rl_sensors
from pic4rl.pic4rl_sensors import OdomSensor, pose_2_xyyaw
from pic4rl.pic4rl_sensors import CmdVelInfo
from pic4rl.pic4rl_sensors import LaserScanSensor, clean_laserscan, laserscan_2_list, laserscan_2_n_points_list
from pic4rl.pic4rl_robots import MobileRobotState
from pic4rl.pic4rl_sensors import s7b3State

from pic4rl.pic4rl_sensors_class import Sensors
from pic4rl.pic4rl_env import Pic4rl

from gym import spaces

import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer


class Pic4rlRobot(Sensors, MobileRobotState, Pic4rl):
	def __init__(self):
		Pic4rl.__init__(self)
		Sensors.__init__(self, 
						generic_laser_scan_sensor = True,
						odometry_sensor = True)
		MobileRobotState.__init__(self)

		action =[
		[-0.5, 0.5], # x_speed 
		#[-0.5, 0.5], # y_speed
		[-1, 1] # theta_speed
		]


		low_action = []
		high_action = []
		for i in range(len(action)):
			low_action.append(action[i][0])
			high_action.append(action[i][1])

		low_action = np.array(low_action, dtype=np.float32)
		high_action = np.array(high_action, dtype=np.float32)

		self.action_space = spaces.Box(
			low=low_action,
			high=high_action,
			#shape=(1,),
			dtype=np.float32
		)
		
		"""
		state
		"""
		state =[
		[0., 5.], # goal_distance 
		[-math.pi, math.pi], # goal_angle
		#[-math.pi, math.pi] # yaw
		]
		

		low_state = []
		high_state = []
		for i in range(len(state)):
			low_state.append(state[i][0])
			high_state.append(state[i][1])

		self.low_state = np.array(low_state, dtype=np.float32)
		self.high_state = np.array(high_state, dtype=np.float32)

		self.observation_space = spaces.Box(
			low=self.low_state,
			high=self.high_state,
			dtype=np.float32
		)

      
def main(args=None):
	rclpy.init()
	try:
		pic4rl = Pic4rlRobot()
		#	rclpy.spin()

		parser = Trainer.get_argument()
		parser = DDPG.get_argument(parser)
		args = parser.parse_args()
		policy = DDPG(
			state_shape=pic4rl.observation_space.shape,
			action_dim=pic4rl.action_space.high.size,
			gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
			memory_capacity=10000,
			max_action=pic4rl.action_space.high,
			lr_actor = 0.00025,
			lr_critic = 0.00025,
			batch_size=64,
			n_warmup=500)
		trainer = Trainer(policy, pic4rl, args, test_env=None)
		trainer()
	finally:
		pic4rl.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()
