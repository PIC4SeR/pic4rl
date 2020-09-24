#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert


activate_this_file = '/home/enricosutera/envs/tf2/bin/activate_this.py'

exec(compile(open(activate_this_file, "rb").read(), activate_this_file, 'exec'), dict(__file__=activate_this_file))


import os
import random
import sys
import time

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from omnirob_msgs.srv import State, Reset, Step

import numpy as np

import collections
#import keras
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.initializers import RandomUniform
import json
import numpy as np
import os
import sys
import random
import math
import time

from gym import spaces

import gym
from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer
from pic4rl.pic4rl_environment import Pic4rlEnvironment

class Pic4rlTraining(Pic4rlEnvironment):
	def __init__(self):
		super().__init__()
		#rclpy.logging.set_logger_level('omnirob_rl_agent', 20)
		#rclpy.logging.set_logger_level('omnirob_rl_environment', 10)

		"""************************************************************
		** Initialise ROS publishers and subscribers
		************************************************************"""
		qos = QoSProfile(depth=10)

		#self.env = OmnirobRlEnvironment()

		self.instanciate_agent()

	def instanciate_agent(self):
		"""
		ACTION AND OBSERVATION SPACES settings for DDPG
		"""
		"""
		actions
		"""
		action =[
		[-0.5, 0.5], # x_speed 
		[-0.5, 0.5], # y_speed
		[-1, 1] # theta_speed
		]


		low_action = []
		high_action = []
		for i in range(len(action)):
			low_action.append(action[i][0])
			high_action.append(action[i][1])

		low_action = np.array(low_action, dtype=np.float32)
		high_action = np.array(high_action, dtype=np.float32)
		"""	
		max_x_speed = 0.5
		min_x_speed = -0.5
		max_y_speed = 0.5
		min_y_speed = -0.5
		max_z_speed = 1
		min_z_speed = -1


		low_action = np.array(
			[min_x_speed, min_y_speed], dtype=np.float32
		)
		high_action = np.array(
			[max_x_speed, max_y_speed], dtype=np.float32
		)
		"""
		#low_action = min_ang_speed
		#high_action = max_ang_speed

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
		[-math.pi, math.pi] # yaw
		]


		low_state = []
		high_state = []
		for i in range(len(state)):
			low_state.append(state[i][0])
			high_state.append(state[i][1])

		self.low_state = np.array(low_state, dtype=np.float32)
		self.high_state = np.array(high_state, dtype=np.float32)
		"""
		max_goal_distance = 5.
		min_goal_distance = 0.
		max_goal_angle = math.pi
		min_goal_angle = -math.pi
		
		self.low_state = np.array(
			[min_goal_distance, min_goal_angle], dtype=np.float32
		)
		self.high_state = np.array(
			[max_goal_distance, max_goal_angle], dtype=np.float32
		)
		"""
		self.observation_space = spaces.Box(
			low=self.low_state,
			high=self.high_state,
			dtype=np.float32
		)

		parser = Trainer.get_argument()
		parser = DDPG.get_argument(parser)
		args = parser.parse_args()
		policy = DDPG(
			state_shape=self.observation_space.shape,
			action_dim=self.action_space.high.size,
			gpu=-1,  # Run on CPU. If you want to run on GPU, specify GPU number
			memory_capacity=10000,
			max_action=self.action_space.high,
			lr_actor = 0.00025,
			lr_critic = 0.00025,
			batch_size=64,
			n_warmup=500)
		trainer = Trainer(policy, self, args, test_env=None)
		trainer()





def main(args=None):
	rclpy.init()
	pic4rl_training= Pic4rlTraining()

	pic4rl_training.get_logger().info('Node spinning ...')
	rclpy.spin(pic4rl_training)

	pic4rl_training.destroy()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
