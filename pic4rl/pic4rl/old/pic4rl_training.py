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

#from pic4rl.pic4rl_environment import Pic4rlEnvironment
#from pic4rl.pic4rl_turtlebot3_burger import Pic4rlTurtleBot3
from pic4rl.pic4rl_tb3_burger_lidar import Pic4rlTurtleBot3

from tf2rl.experiments.trainer import Trainer
from tf2rl.algos.ddpg import DDPG

class Pic4rlTraining(Pic4rlTurtleBot3):
	def __init__(self):
		#self.env = Pic4rlTurtleBot3()
		super().__init__()
		print(self.observation_space.shape)

		self.instanciate_agent()

	def instanciate_agent(self):
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

	"""pic4rl_training.get_logger().info('Node spinning ...')
	rclpy.spin(pic4rl_training)

	pic4rl_training.destroy()
	rclpy.shutdown()"""

if __name__ == '__main__':
	main()
