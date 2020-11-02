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

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
	try:
	# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
	# Memory growth must be set before GPUs have been initialized
		print(e)

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step

import collections

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal, HeUniform, GlorotUniform
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K

import json
import numpy as np
import random
import sys
import time
import math
import gc

from pic4rl.pic4rl_environment import Pic4rlEnvironment

MAX_LIN_SPEED = 0.2
MAX_ANG_SPEED = 1
DESIRED_CTRL_HZ = 1

class Pic4rlTraining(Pic4rlEnvironment):
    def __init__(self):
        super().__init__()
        #rclpy.logging.set_logger_level('omnirob_rl_agent', 20)
        #rclpy.logging.set_logger_level('omnirob_rl_environment', 10)

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        self.env = Pic4rlEnvironment()
        #self.stage = int(stage)
	
        #self.avg_cmd_vel = [0.2,int(0)]
        #self.evalutate_Hz(init=True)

        # State size and action size
        self.state_size = 38
        self.action_size = 2 #linear velocity, angular velocity
        self.episode_size = 10
        #self.state_size = 3 #goal distance, goal angle, lidar points
        #self.height = 60
        #self.width = 80
        # DDPG hyperparameter
        # self.tau = 0.001
        # self.discount_factor = 0.99
        # self.learning_rate = 0.00025
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.998
        # self.epsilon_min = 0.05
        # self.batch_size = 64
        # self.train_start = 64
        # self.update_target_model_start = 128
        # self.score_list = []

        # Replay memory
        #self.memory = collections.deque(maxlen=1000000)

        # Build actor and critic models and target models
        self.actor_model, self.actor_optimizer = self.build_actor()

       # Load saved models
        self.load_model = True
        self.load_episode = 0
        self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_dir_path = self.model_dir_path.replace(
            '/pic4rl/pic4rl/pic4rl',
            '/pic4rl/pic4rl/models/agent_model')
        self.results_path = '/home/mauromartini/mauro_ws/test'

        self.actor_model_path = os.path.join(
            self.model_dir_path,
            'actor_lidar_episode2620'+'.h5')

        if self.load_model:
            self.actor_model.set_weights(load_model(self.actor_model_path).get_weights()) 
          
        """************************************************************
        ** Initialise ROS clients
        ************************************************************"""
        # Initialise clients
        #self.Ddpg_com_client = self.create_client(Ddpg, 'Ddpg_com')
        """************************************************************
        ** Start process
        ************************************************************"""
        self.process()

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def process(self):
        global_step = 0

        for episode in range(1, self.episode_size):
            global_step += 1
            local_step = 0

            done = False
            init = True
            score = 0

            # Reset environment
            next_state, goal_pose_x, goal_pose_y = self.env.reset(episode)
            goal_pose = [goal_pose_x, goal_pose_y]
            time_start = time.time()
            #state = np.array(state, dtype=np.float32)
            #print('Goal distance, goal angle, lidar points', state)

            while not done:
                #time_step = time.time()
                local_step += 1

                state = next_state
                action = self.get_action(state)
                if np.isnan(action[1]):
                    print("Action:", action)
                    action = np.array([0.0, 0.0])

                next_state, reward, done, info, total_distance, goal_distance = self.env.step(action)
                score += reward

                # Save <s, a, r, s'> samples
                if local_step > 1:
                    if done:
                        time_path = time.time() - time_start
                        print(
                            "Test episode:", episode,
                            "score:", score,
                            "Time:", time_path,
                            "Path:", total_distance,
                            "Final distance:", goal_distance)
                            #"memory length:", len(self.memory),
                            #"epsilon:", self.epsilon)
			                #"avg Hz:", 1/self.avg_cmd_vel[0])

                        param_keys = ['goal_pose_x', 'score', 'time', 'path', 'final_distance']
                        param_values = [goal_pose, score, time_path, total_distance, goal_distance]
                        param_dictionary = dict(zip(param_keys, param_values))

                #time_step_end = time.time() - time_step
                #print('time for local step:', time_step_end)
                # While loop rate
                #current_hz = 1/self.avg_cmd_vel[0]
                #time.sleep(max((current_hz-DESIRED_CTRL_HZ),0))

            #Save score
            with open(os.path.join(self.results_path,'test4_lidar_episode'+str(episode)+'.json'), 'w') as outfile:
                json.dump(param_dictionary, outfile)

    def evalutate_Hz(self, init = False):
        if init:
                self.start = time.time()
        else:
                end = time.time() 
                delta = end - self.start

                if  delta<=3:
                        self.avg_cmd_vel[1]+=1

                        self.avg_cmd_vel[0] = (self.avg_cmd_vel[0]*(self.avg_cmd_vel[1]-1)\
								 + delta)\
								/self.avg_cmd_vel[1]
                self.start = end

    def build_actor(self):

        state_input = Input(shape=(self.state_size,))
        h1 = Dense(512, activation='relu')(state_input)
        h2 = Dense(256, activation='relu')(h1)
        out1 = Dense(256, activation='relu')(h2)
        #out1 = Dropout(0.2)(out1)

        Linear_velocity = Dense(1, activation = 'sigmoid')(out1)*MAX_LIN_SPEED
        Angular_velocity = Dense(1, activation = 'tanh')(out1)*MAX_ANG_SPEED
        output = concatenate([Linear_velocity,Angular_velocity])

        model = Model(inputs=[state_input], outputs=[output])
        adam = Adam(lr= 0.0001)
        #model.summary()

        return model, adam

    def build_actor_camera(self):
        goal_input = Input(shape=(2,))
        depth_image_input = Input(shape=(self.height, self.width, 1,))
        initializer = HeUniform()
        #last_init = tf.keras.initializers.glorot_uniform()
        #depth_image_norm = BatchNormalization()(depth_image_input)
        #goal_norm = BatchNormalization()(goal_input)

        c1 = Conv2D(32, 3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(depth_image_input)
        c2 = Conv2D(32, 3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c1)
        c2p = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c2)
        c3 = Conv2D(64, 3, strides=(2, 2), activation='relu', kernel_initializer = initializer)(c2p)
        c4 = Conv2D(64, 3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c3)
        #c5 = Conv2D(128, 3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c4)
        h0 = GlobalAveragePooling2D()(c4)

        fc1 = Dense(128, activation='relu', kernel_initializer = initializer)(h0)
        fc2 = Dense(64, activation='relu', kernel_initializer = initializer)(fc1)
        #fc2s = Dense(32, activation='relu', kernel_initializer = initializer)(fc1s)
        conc1 = concatenate([goal_input,fc2])
        out1 = Dense(128, activation='relu', kernel_initializer = initializer)(conc1)

        Linear_velocity = Dense(1, activation = 'sigmoid')(out1)*MAX_LIN_SPEED
        Angular_velocity = Dense(1, activation = 'tanh')(out1)*MAX_ANG_SPEED
        output = concatenate([Linear_velocity,Angular_velocity])

        model = Model(inputs=[goal_input, depth_image_input], outputs=[output], name='Actor')
        adam = Adam(lr= 0.0001)
        model.summary()

        return model, adam

    def get_action(self, state):
        #LIDAR
        pred_action = self.actor_model(state.reshape(1, len(state)))
        return [pred_action[0][0], pred_action[0][1]]

        #CAMERA
        #goal = tf.reshape(state[0], [1,2])
        #depth_image = tf.reshape(state[1], [1,self.height, self.width])
        #pred_action = self.actor_model([goal, depth_image])
        #return [pred_action[0][0], pred_action[0][1]]

def main(args=None):
    rclpy.init()
    pic4rl_training= Pic4rlTraining()

    pic4rl_training.get_logger().info('Node spinning ...')
    rclpy.spin(pic4rl_training)

    pic4rl_training.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
