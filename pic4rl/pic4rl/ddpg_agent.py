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

import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Ddpg

MAX_LIN_SPEED = 0.2
MAX_ANG_SPEED = 1
DESIRED_CTRL_HZ = 3

class DDPGAgent(Node):
    def __init__(self, stage):
        super().__init__('ddpg_agent')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        # Stage
        self.stage = int(stage)
	
        self.avg_cmd_vel = [0.3,int(0)]
        self.evalutate_Hz(init=True)

        # State size and action size
        self.state_size = 4 #goal distance, goal angle, lidar points
        self.action_size = 2 #linear velocity, angular velocity
        self.action_lin_low = 0
        self.action_lin_high = 1
        self.action_ang_low = -1
        self.action_ang_high = 1
        self.episode_size = 1000

        # DDPG hyperparameter
        self.tau = 0.001
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 86
        self.update_target_model_start = 128

        # Replay memory
        self.memory = collections.deque(maxlen=1000000)

        # Build actor and critic models and target models
        self.actor_model, self.actor_optimizer = self.build_actor()
        self.critic_model = self.build_critic()
        self.target_actor_model, _ = self.build_actor()
        self.target_critic_model = self.build_critic()

       # Load saved models
        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_dir_path = self.model_dir_path.replace(
            'turtlebot3_ddpg/ddpg_agent',
            'model')
        self.actor_model_path = os.path.join(
            self.model_dir_path,
            'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')
        self.critic_model_path = os.path.join(
            self.model_dir_path,
            'critic_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')

        if self.load_model:
            self.actor_model.set_weights(load_model(self.actor_model_path).get_weights())
            with open(os.path.join(
                    self.model_dir_path,
                    'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

            self.critic_model.set_weights(load_model(self.critic_model_path).get_weights())
            with open(os.path.join(
                    self.model_dir_path,
                    'critic_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')                

        """************************************************************
        ** Initialise ROS clients
        ************************************************************"""
        # Initialise clients
        self.Ddpg_com_client = self.create_client(Ddpg, 'Ddpg_com')
        """************************************************************
        ** Start process
        ************************************************************"""
        self.process()

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def process(self):
        global_step = 0

        for episode in range(self.load_episode+1, self.episode_size):
            global_step += 1
            local_step = 0

            if global_step == self.train_start+1:
                print('Start training models, global step:', global_step)

            state = list()
            next_state = list()
            done = False
            init = True
            score = 0

            # Reset ddpg environment
            time.sleep(1.0)

            while not done:
                local_step += 1

                # Action based on the current state
                if local_step == 1:
                    action = np.array([0.0, 0.0])

                else:
                    state = next_state
                    action = self.get_action(state)
                    if np.isnan(action[1]):
                        print("Action:", action)
                        action = np.array([0.0, 0.0])
                    #print("Action:", action)
                    #print("Action size:", action.shape)

                # Send action and receive next state and reward
                req = Ddpg.Request()
                lin_vel = action[0]
                ang_vel = action[1]
                req.linear_velocity = float(lin_vel)
                req.angular_velocity = float(ang_vel)
                req.init = init
                while not self.Ddpg_com_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('service not available, waiting again...')
		
                self.evalutate_Hz()
                future = self.Ddpg_com_client.call_async(req)

                while rclpy.ok():
                    rclpy.spin_once(self)
                    if future.done():
                        if future.result() is not None:
                            # Next state and reward
                            next_state = future.result().state
                            if np.isnan(sum(next_state)):
                                print("next_state ",next_state)
                                raise ValueError("received wrong next_state")
                            reward = future.result().reward
                            if np.isnan(reward):
                                print("reward ",reward)
                                raise ValueError("received wrong reward")
                            done = future.result().done
                            score += reward
                            init = False
                        else:
                            self.get_logger().error(
                                'Exception while calling service: {0}'.format(future.exception()))
                        break

                # Save <s, a, r, s'> samples
                if local_step > 1:
                    self.append_sample(state, action, next_state, reward, done)

                    # Train model
                    if global_step > self.update_target_model_start:
                        #print('Update target model, global step:', global_step)
                        self.train_model(True)
                    elif global_step > self.train_start:
                        #print('Training models, global step:', global_step)
                        #print('time start train:', time.time())
                        self.train_model()
                        #print('time end train:', time.time())

                    if done:
                        # Update neural network
                        self.update_target_model()
                        #self.target_actor_model = self.update_target_model_soft(self.actor_model, self.target_actor_model, self.tau)
                        #self.target_critic_model = self.update_target_model_soft(self.critic_model, self.target_critic_model, self.tau)

                        print(
                            "Episode:", episode,
                            "score:", score,
                            "memory length:", len(self.memory),
                            "epsilon:", self.epsilon,
			                "avg Hz:", 1/self.avg_cmd_vel[0])

                        param_keys = ['epsilon']
                        param_values = [self.epsilon]
                        param_dictionary = dict(zip(param_keys, param_values))

                # While loop rate
                current_hz = 1/self.avg_cmd_vel[0]
                time.sleep(max((current_hz-DESIRED_CTRL_HZ),0))

            # Update result and save model every 10 episodes
            if episode % 10 == 0:
                self.actor_model_path = os.path.join(
                    self.model_dir_path,
                    'actor_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
                self.actor_model.save(self.actor_model_path)
                with open(os.path.join(
                    self.model_dir_path,
                        'actor_stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
                    json.dump(param_dictionary, outfile)

                self.critic_model_path = os.path.join(
                    self.model_dir_path,
                    'critic_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
                self.critic_model.save(self.critic_model_path)
                with open(os.path.join(
                    self.model_dir_path,
                        'critic_stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
                    json.dump(param_dictionary, outfile)

            # Epsilon (exploration policy)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

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
        model.summary()

        return model, adam

    def build_critic(self):
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
    
    def update_target_model(self):
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())

    def update_target_model_soft(self, online, target, tau):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in range(len(init_weights)):
            weights.append(tau * init_weights[i] + (1 - tau) * update_weights[i])
        target.set_weights(weights)
        return target

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
                rnd_action = [random.random()*MAX_LIN_SPEED, (random.random()*2-1)*MAX_ANG_SPEED]
                #print("rnd_action",rnd_action)
                return rnd_action
        else:
                state = np.asarray(state, dtype= np.float32)
                print("state in prediction:",state)
                pred_action = self.actor_model(state.reshape(1, len(state)))
                print("pred_action", pred_action)
                return [pred_action[0][0], pred_action[0][1]]


    def append_sample(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))


    def train_model(self, target_train_start=False):
 
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

            self.train_critic(states, actions, next_states, rewards, dones, target_train_start)
            self.train_actor(states, actions, next_states, rewards, dones)


    def train_critic(self, states, actions, next_states, rewards, dones, target_train_start):

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
 
    def train_actor(self, states, actions, next_states, rewards, dones):
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


def main(args=sys.argv[1]):
    rclpy.init(args=args)
    ddpg_agent = DDPGAgent(args)
    rclpy.spin(ddpg_agent)

    ddpg_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()