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
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
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
from tensorflow.keras.initializers import RandomUniform, glorot_normal
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import json
import numpy as np
import random
import sys
import time
import math

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
        self.state_size = 3 #goal distance, goal angle, lidar points
        self.action_size = 2 #linear velocity, angular velocity
        self.height = 224
        self.width = 224
        self.image_shape = [224,224,3]
        self.episode_size = 5000

        # DDPG hyperparameter
        self.tau = 0.001
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.05
        self.batch_size = 32
        self.train_start = 3
        self.update_target_model_start = 5

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
        # self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.model_dir_path = self.model_dir_path.replace(
        #     'turtlebot3_ddpg/ddpg_agent',
        #     'model')
        # self.actor_model_path = os.path.join(
        #     self.model_dir_path,
        #     'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')
        # self.critic_model_path = os.path.join(
        #     self.model_dir_path,
        #     'critic_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')

        # if self.load_model:
        #     self.actor_model.set_weights(load_model(self.actor_model_path).get_weights())
        #     with open(os.path.join(
        #             self.model_dir_path,
        #             'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
        #         param = json.load(outfile)
        #         self.epsilon = param.get('epsilon')

        #     self.critic_model.set_weights(load_model(self.critic_model_path).get_weights())
        #     with open(os.path.join(
        #             self.model_dir_path,
        #             'critic_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
        #         param = json.load(outfile)
        #         self.epsilon = param.get('epsilon')                

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
            state = self.env.reset()
            print('state raw shape: ',state.shape)
            #print(state)

            while not done:
                local_step += 1
                #print('new local step at time: ', time.time())
                # Action based on the current state
                if local_step == 1:
                    action = np.array([0.0, 0.0])

                else:
                    state = next_state
                    action = self.get_action(state)
                    if np.any(np.isnan(action)):
                        print("Action:", action)
                        action = tf.constant([0.0, 0.0])
                    #print("Action:", action)
                    #print("Action size:", action.shape)

                next_state, reward, done, info = self.env.step(action)
                #print('next state:', next_state)
                #print(next_state.shape)
                
                score += reward

                #self.evalutate_Hz()

                # Save <s, a, r, s'> samples
                if local_step > 1:
                    self.append_sample(state, action, next_state, reward, done)

                    # Train model
                    if global_step >= self.update_target_model_start:
                        #print('Update target model, global step:', global_step)
                        time_start = time.time()
                        self.train_model(True)
                        time_diff = time.time() - time_start
                        print('Total time for training:', time_diff)
                    elif global_step >= self.train_start:
                        print('Start training. Global step:', global_step)
                        time_start = time.time()
                        self.train_model()
                        time_diff = time.time() - time_start
                        print('Total time for training:', time_diff)

                    if done:
                        # Update neural network
                        #self.update_target_model()
                        self.target_actor_model = self.update_target_model_soft(self.actor_model, self.target_actor_model, self.tau)
                        self.target_critic_model = self.update_target_model_soft(self.critic_model, self.target_critic_model, self.tau)

                        print(
                            "Episode:", episode,
                            "score:", score,
                            "memory length:", len(self.memory),
                            "epsilon:", self.epsilon)
			                #"avg Hz:", 1/self.avg_cmd_vel[0])

                        param_keys = ['epsilon']
                        param_values = [self.epsilon]
                        param_dictionary = dict(zip(param_keys, param_values))

                # While loop rate
                #current_hz = 1/self.avg_cmd_vel[0]
                #time.sleep(max((current_hz-DESIRED_CTRL_HZ),0))

            # Update result and save model every 10 episodes
            # if episode % 10 == 0:
            #     self.actor_model_path = os.path.join(
            #         self.model_dir_path,
            #         'actor_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
            #     self.actor_model.save(self.actor_model_path)
            #     with open(os.path.join(
            #         self.model_dir_path,
            #             'actor_stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
            #         json.dump(param_dictionary, outfile)

            #     self.critic_model_path = os.path.join(
            #         self.model_dir_path,
            #         'critic_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
            #     self.critic_model.save(self.critic_model_path)
            #     with open(os.path.join(
            #         self.model_dir_path,
            #             'critic_stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
            #         json.dump(param_dictionary, outfile)

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
        state_input = Input(shape=(self.width, self.height,3,))

        c1 = Conv2D(96,3, strides=(2, 2), activation='relu')(state_input)
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c1)
        c2 = Conv2D(128,3,strides=(2, 2),activation='relu')(c1)
        c2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c2)
        c3 = Conv2D(128,5,strides=(1, 1),activation='relu')(c2)
        h0 = GlobalAveragePooling2D()(c3)

        h1 = Dense(256, activation='relu')(h0)
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
        state_input = Input(shape=(self.width, self.height,3,))      
        actions_input = Input(shape=(self.action_size,))

        c1 = Conv2D(96,3,strides=(2, 2), activation='relu')(state_input)
        c1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c1)
        c2 = Conv2D(128,3,strides=(2, 2), activation='relu')(c1)
        c2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c2)
        c3 = Conv2D(128,5,strides=(1, 1), activation='relu')(c2)
        h0 = GlobalAveragePooling2D()(c3)

        h_state = Dense(256, activation='relu')(h0)
        h_action = Dense(64, activation='relu')(actions_input)
        concatenated = concatenate([h_state, h_action])
        concat_h1 = Dense(256, activation='relu')(concatenated)
        concat_h2 = Dense(128, activation='relu')(concat_h1)
        #concat_h2 = Dropout(0.2)(concat_h2)

        output = Dense(1, activation='linear')(concat_h2)
        model = Model(inputs=[state_input, actions_input], outputs=[output])
        adam  = Adam(lr=0.0008)
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
                #state = np.asarray(state, dtype= np.float32)
                state = tf.reshape(state, [1,self.width,self.height,3])
                #print("state in prediction:",state.shape)
                #pred_action = self.actor_model(state.reshape(1, 224,224,3))
                pred_action = self.actor_model(state)
                print("pred_action", pred_action)
                return [pred_action[0][0], pred_action[0][1]]
                #return pred_action

    def append_sample(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))


    def train_model(self, target_train_start=False):
            time_check = time.time()
            mini_batch = random.sample(self.memory, self.batch_size)
            # states = []
            #actions = []
            # next_states = []
            #rewards = []
            #dones = []            
            # for i in range(self.batch_size):
            #     state = np.asarray(mini_batch[i][0], dtype= np.float32)
            #     action = np.asarray(mini_batch[i][1], dtype= np.float32)
            #     next_state = np.asarray(mini_batch[i][2], dtype= np.float32)
            #     reward = np.asarray(mini_batch[i][3], dtype= np.float32)
            #     done = np.asarray(mini_batch[i][4], dtype= np.float32)

            #     states.append(state)
            #     actions.append(action)
            #     next_states.append(next_state)
            #     rewards.append(reward)
            #     dones.append(done)

            for i in range(self.batch_size):
                if i == 0:
                    tmp_state = tf.expand_dims(mini_batch[i][0],0)
                    states = tmp_state
                    tmp_next_state = tf.expand_dims(mini_batch[i][2],0)
                    next_states = tmp_next_state
                    tmp_action = tf.expand_dims(mini_batch[i][1],0)
                    actions = tmp_action
                    tmp_reward = tf.constant(mini_batch[i][3], dtype= tf.float32)
                    tmp_reward = tf.expand_dims(tmp_reward,0)
                    rewards = tmp_reward
                    tmp_done = tf.constant(mini_batch[i][4], dtype= tf.float32)
                    tmp_done = tf.expand_dims(tmp_done,0)
                    dones = tmp_done

                else:
                    tmp_state = tf.expand_dims(mini_batch[i][0], 0)   
                    states = tf.concat([states, tmp_state], axis=0)
                    tmp_next_state = tf.expand_dims(mini_batch[i][2], 0)   
                    next_states = tf.concat([next_states, tmp_next_state], axis=0)
                    tmp_action = tf.expand_dims(mini_batch[i][1], 0)   
                    actions = tf.concat([actions, tmp_action], axis=0)
                    tmp_reward = tf.constant(mini_batch[i][3], dtype= tf.float32)
                    tmp_reward = tf.expand_dims(tmp_reward,0)
                    rewards = tf.concat([rewards, tmp_reward], axis=0)
                    tmp_done = tf.constant(mini_batch[i][4], dtype= tf.float32)
                    tmp_done = tf.expand_dims(tmp_done,0)
                    dones = tf.concat([dones, tmp_done], axis=0)

            print('state shape', states.shape)
            print('actions shape', actions.shape)
            print('rewards shape', rewards.shape)
            print('dones shape', dones.shape)

            print('time to set batch', time.time()-time_check)
            time_check = time.time()
            self.train_critic(states, actions, next_states, rewards, dones, target_train_start)
            print('time to train critic', time.time()-time_check)
            time_check = time.time()
            self.train_actor(states, actions, next_states, rewards, dones)
            print('time to train actor', time.time()-time_check)

    #@tf.function
    def train_critic(self, states, actions, next_states, rewards, dones, target_train_start):
        time_check = time.time()
        error = False

        try:
            if not target_train_start:
                target_actions = self.actor_model([next_states])
                #target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])
            else:
                target_actions = self.target_actor_model([next_states])
                #target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])

        except:
            #print("next_states ",next_states)
            #print("tensor next_states",next_states)
            error = True
        if error:
            print('Error in train critic, target action')
            target_actions = self.target_actor_model([next_states])

        print('time for target actions', time.time()-time_check)
        #print("target action shape", target_actions.shape)

        time_check = time.time()
        if not target_train_start:
                target_q_values = self.critic_model([next_states, target_actions])
                #target_q_values = tf.reshape(target_q_values, [self.batch_size, None])

        else:
                target_q_values = self.target_critic_model([next_states, target_actions])
                #target_q_values = tf.reshape(target_q_values, [self.batch_size, None])

        print('time for target q values', time.time()-time_check)
        time_check = time.time()
        #dones = tf.constant(dones)
        #rewards = tf.constant(rewards, shape=[self.batch_size,None])
        #print("rewards.shape ",rewards.shape)
        #print("rewards ",rewards)
        #print("target_q_values.shape ", target_q_values.shape)
        #print("dones.shape ", dones.shape)
        ds = tf.math.subtract(tf.ones(shape=dones.shape),dones)
        disc = tf.math.multiply(self.discount_factor,ds)
        disc_targ= tf.math.multiply(target_q_values,disc)
        targets = tf.math.add(rewards,disc_targ)
        #targets = rewards + target_q_values*self.discount_factor*(tf.ones(shape=dones.shape)-dones)
        #print("targets.shape ",targets.shape)
        #print("targets",targets)

        #tf_states = tf.convert_to_tensor(states)
        #actions = np.stack(actions)
        #actions = tf.convert_to_tensor(actions)
        #targets = tf.convert_to_tensor(targets)
        #print("states tensor shape ", states.shape)
        #print("actions tensor shape ", actions.shape)
        #print("action tensor ex", actions[0])
        #print("target tensor ex", targets[0])
        print('time for targets', time.time()-time_check)
        time_check = time.time()
        #inputs must be a list of tensors since we have a multiple inputs NN
        self.critic_model.train_on_batch([states, actions], targets)
        print('time for critic train on batch', time.time()-time_check)
    
    @tf.function
    def train_actor(self, states, actions, next_states, rewards, dones):
        time_check = time.time()
        with tf.GradientTape() as tape:
            a = self.actor_model([states])
            tape.watch(a)
            q = self.critic_model([states, a])
        dq_da = tape.gradient(q, a)
        #print('Action gradient dq_qa: ', dq_da)
        print('time for actor grad tape 1', time.time()-time_check)

        time_check = time.time()
        with tf.GradientTape() as tape:
            a = self.actor_model([states])
            theta = self.actor_model.trainable_variables
            tape.watch(theta)
        da_dtheta = tape.gradient(a, theta, output_gradients= -dq_da)
        #print('Actor loss da_dtheta: ', da_dtheta)
        actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), da_dtheta))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))
        #self.actor_optimizer.apply_gradients(zip(da_dtheta, self.actor_model.trainable_variables))
        print('time for apply actor gradient', time.time()-time_check)


def main(args=None):
    rclpy.init()
    pic4rl_training= Pic4rlTraining()

    pic4rl_training.get_logger().info('Node spinning ...')
    rclpy.spin(pic4rl_training)

    pic4rl_training.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
