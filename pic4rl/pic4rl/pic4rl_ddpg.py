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
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import json
import numpy as np
import os
import random
import sys
import time
import math

from pic4rl.pic4rl_environment import Pic4rlEnvironment
#from turtlebot3_msgs.srv import Ddpg

class Pic4rlDDPGAgent():
    """####################


    ####################"""
    def __init__(self, env):
        """####################
        # Agent (training) settings
        ####################"""
        self.tau = 0.001    # Target networks soft update parameter
        self.epsilon = 1.0  # Epsilon-greedy exploration policy starting value
        self.epsilon_decay = 0.996  # Epsilon-greedy exploration policy decay parameter
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 86
        self.update_target_model_start = 128
        self.max_episode = 1000

        """####################
        # Agent (training) variable
        ####################"""
        self.local_step = None      # Episode step (is reset after each episode)
        self.global_step = None     # Global step (is not reset after each episode)
        self.episode = None         # Current episode
        self.score = []


        """####################
        # Sub-components (classes)
        ####################"""
        self.memory = collections.deque(maxlen=1000000)
        self.actor_model, self.actor_optimizer = self._build_actor()
        self.critic_model = self._build_critic()
        self.target_actor_model, _ = self._build_actor()
        self.target_critic_model = self._build_critic()

        """####################
        # Other settings
        ####################"""

        # Save/load settings to be added

        """####################
        # INITIALIZATION
        ####################"""

    def _fit_env():
        # This function should provide instruction to set inputs and outputs of the agent

    def training():
        self.global_step = 0

        for episode in range(self.max_episode):
            self.score.append(0)
            self.local_step = 0
            observation = self._reset()

            while True:
                action = self.get_action(observation)
                next_observation, reward, done, info = self._step(action)

                self.score[:] += reward     #Update episode score (last element of score list)
                self._append_sample(observation, action, next_observation, reward, done, info)
                self._train_model()

                if done:
                    self.update_target_model()
                    print(
                            "Episode:", episode,
                            "score:", self.score[:],
                            "memory length:", len(self.memory),
                            "epsilon:", self.epsilon)

    def _step():

    def _append_sample():
        # in previous version a self.local_step > 1 condition was present to allow the saving
        self.memory.append((state, action, next_state, reward, done))

    def _train_model(self, target_train_start=False):
 
            mini_batch = random.sample(self.memory, self.batch_size)
            states = []
            actions = []
            next_states = []
            rewards = []
            dones = []            
            for i in range(self.batch_size):
                state = np.asarray(mini_batch[i][0], dtype= np.float32)
                action = np.asarray(mini_batch[i][1], dtype= np.float32)
                next_state = np.asarray(mini_batch[i][2], dtype= np.float32)
                reward = np.asarray(mini_batch[i][3], dtype= np.float32)
                done = np.asarray(mini_batch[i][4], dtype= np.float32)

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

            self._train_critic(states, actions, next_states, rewards, dones, target_train_start)
            self._train_actor(states, actions, next_states, rewards, dones)

    def _train_critic(self, states, actions, next_states, rewards, dones, target_train_start):

        tf_next_states = tf.convert_to_tensor(next_states)
        error = False

        try:
            if not target_train_start:
                target_actions = self.actor_model.predict([tf_next_states])
                target_actions = np.asarray(target_actions, dtype= np.float32).reshape(self.batch_size, self.action_size)
            else:
                target_actions = self.target_actor_model.predict([tf_next_states])
                target_actions = np.asarray(target_actions, dtype= np.float32).reshape(self.batch_size, self.action_size)

        except:
            #print("next_states ",next_states)
            #print("tensored next_states",tf_next_states)
            error = True
        if error:
            print('Error in train critic, target action')
            target_actions = self.target_actor_model.predict([tf_next_states])

        #print("target action ex", target_actions[0])

        if not target_train_start:
                target_q_values = self.critic_model.predict([tf_next_states, target_actions])
                target_q_values = np.asarray(target_q_values, dtype= np.float32).reshape(self.batch_size,)

        else:
                target_q_values = self.target_critic_model.predict([tf_next_states, target_actions])
                target_q_values = np.asarray(target_q_values, dtype= np.float32).reshape(self.batch_size,)

        dones = np.array(dones)
        rewards = np.array(rewards).reshape(self.batch_size,)
        #print("rewards.shape ",rewards.shape)
        #print("rewards ",rewards)
        #print("target_q_values.shape ", target_q_values.shape)
        #print("dones.shape ", dones.shape)

        targets = rewards + target_q_values*self.discount_factor*(np.ones(shape=dones.shape) - dones)
        #print("targets.shape ",targets.shape)
        #print("targets ex",targets[0])

        tf_states = tf.convert_to_tensor(states)
        actions = np.stack(actions)
        tf_actions = tf.convert_to_tensor(actions)
        tf_targets = tf.convert_to_tensor(targets)
        #print("states tensor shape ", tf_states.shape)
        #print("actions tensor shape ", tf_actions.shape)
        #print("action tensor ex", tf_actions[0])
        #print("target tensor ex", tf_targets[0])

        #inputs must be a list of tensors since we have a multiple inputs NN
        self.critic_model.train_on_batch([tf_states, tf_actions], tf_targets)
 
    def _train_actor(self, states, actions, next_states, rewards, dones):
        tf_states = tf.convert_to_tensor(states)
        with tf.GradientTape() as tape:
            a = self.actor_model([tf_states])
            tape.watch(a)
            q = self.critic_model([tf_states, a])
        dq_da = tape.gradient(q, a)
        #print('Action gradient dq_qa: ', dq_da)
    

        with tf.GradientTape() as tape:
            a = self.actor_model([tf_states])
            theta = self.actor_model.trainable_variables
            tape.watch(theta)
        da_dtheta = tape.gradient(a, theta, output_gradients= -dq_da)
        #print('Actor loss da_dtheta: ', da_dtheta)
        actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), da_dtheta))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))
        #self.actor_optimizer.apply_gradients(zip(da_dtheta, self.actor_model.trainable_variables))

    def _build_actor(self):

        state_input = Input(shape=(self.state_size,))
        h1 = Dense(512, activation='relu')(state_input)
        h2 = Dense(256, activation='relu')(h1)
        out1 = Dense(256, activation='relu')(h2)
        #out1 = Dropout(0.2)(out1)

        Linear_velocity = Dense(1, activation = 'sigmoid')(out1)*MAX_LIN_SPEED
        Angular_velocity = Dense(1, activation = 'tanh')(out1)*MAX_ANG_SPEED
        output = concatenate([Linear_velocity,Angular_velocity])

        model = Model(inputs=[state_input], outputs=[output])
        adam = Adam(lr= 0.00025)
        model.summary()

        return model, adam

    def _build_critic(self):
        state_input = Input(shape=(self.state_size,))      
        actions_input = Input(shape=(self.action_size,))

        h_state = Dense(256, activation='relu')(state_input)
        h_action = Dense(64, activation='relu')(actions_input)
        concatenated = concatenate([h_state, h_action])
        concat_h1 = Dense(256, activation='relu')(concatenated)
        concat_h2 = Dense(128, activation='relu')(concat_h1)
        #concat_h2 = Dropout(0.2)(concat_h2)

        output = Dense(1, activation='linear')(concat_h2)
        model = Model(inputs=[state_input, actions_input], outputs=[output])
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        model.summary()

        return model
