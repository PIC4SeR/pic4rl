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

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

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

#with tf.device('/CPU:0'):
MAX_LIN_SPEED = 0.8
MAX_ANG_SPEED = 2
DESIRED_CTRL_HZ = 5

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
        #self.stage = 1
	
        #self.avg_cmd_vel = [0.4,int(0)]
        #self.evalutate_Hz(init=True)

        # State size and action size
        self.state_size = 3 #goal distance, goal angle, lidar points
        self.action_size = 2 #linear velocity, angular velocity
        self.height = 60
        self.width = 80
        self.episode_size = 5000

        # DDPG hyperparameter
        self.tau = 0.01
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.use_target_model = 128
        self.score_list = []

        # Replay memory
        self.memory = collections.deque(maxlen=170000)

        # Build actor and critic models and target models
        self.actor_model, self.actor_optimizer = self.build_actor()
        self.critic_model, self.critic_optimizer = self.build_critic()
        self.target_actor_model, _ = self.build_actor()
        self.target_critic_model, _ = self.build_critic()
        self.update_target_model()

       # Load saved models
        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_dir_path = self.model_dir_path.replace(
            '/pic4rl/pic4rl/pic4rl',
            '/pic4rl/pic4rl/models/agent_model')
        self.results_path = '/home/mauromartini/mauro_ws/scores/camera/rosbot'

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

            self.update_target_model()

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

            #state = list()
            #next_state = list()
            done = False
            init = True
            score = 0
            
            #time_check = time.time()
            # Reset ddpg environment
            state = self.env.reset(episode)
            #print('state raw shape: ',state.shape)
            #print(state)
            #print('time for env reset:', time.time()-time_check)

            while not done:
                local_step += 1
                #time_check = time.time()
                # Action based on the current state
                if local_step == 1:
                    action = np.array([0.0, 0.0])

                else:
                    state = next_state
                    action = self.get_action(state)
                    if np.any(np.isnan(action)):
                        print("Action is nan:", action)
                        action = tf.constant([0.0, 0.0])
                    #print("Action:", action)
                    #print("Action size:", action.shape)
                #print('time for getting action:', time.time()-time_check)

                #time_check = time.time()
                next_state, reward, done, info = self.env.step(action)
                #print('next state:', next_state)
                #print(next_state.shape)
                #print('time for env step:', time.time()-time_check)
                score += reward

                #self.evalutate_Hz()

                # Save <s, a, r, s'> samples
                if local_step > 1:
                    self.append_sample(state, action, next_state, reward, done)

                    # Train model
                    # if global_step >= self.use_target_model:
                    #     #print('Update target model, global step:', global_step)
                    #     #time_start = time.time()
                    #     self.train_model(True)
                    #     #time_diff = time.time() - time_start
                    #     #print('Total time for training:', time_diff)

                    #     # UPDATE TARGET NETWORKS
                    #     #HARD UPDATE
                    #     #self.update_target_model()
                    #     #print('Updating target models')

                    #     #SOFT UPDATE
                    #     #time_start = time.time()
                    #     self.target_actor_model = self.update_target_model_soft(self.actor_model, self.target_actor_model, self.tau)
                    #     self.target_critic_model = self.update_target_model_soft(self.critic_model, self.target_critic_model, self.tau)
                    #     #print('time for target model update:', time.time()-time_check)
                    
                    if global_step >= self.train_start:
                        #print('Start training. Global step:', global_step)
                        #time_check = time.time()
                        self.train_model(True)
                        #print('Total time for training:', time.time() - time_check)

                        # UPDATE TARGET NETWORKS
                        #HARD UPDATE
                        #self.update_target_model()
                        #print('Updating target models')

                        #SOFT UPDATE
                        #time_check= time.time()
                        self.target_actor_model = self.update_target_model_soft(self.actor_model, self.target_actor_model, self.tau)
                        self.target_critic_model = self.update_target_model_soft(self.critic_model, self.target_critic_model, self.tau)
                        #print('time for target model update:', time.time()-time_check)

                    if done:

                        print(
                            "Episode:", episode,
                            "score:", score,
                            "memory length:", len(self.memory),
                            "epsilon:", self.epsilon)
			                #"avg Hz:", 1/self.avg_cmd_vel[0])

                        param_keys = ['epsilon']
                        param_values = [self.epsilon]
                        param_dictionary = dict(zip(param_keys, param_values))
                        self.score_list.append(score)
                        #print(self.score_list)

				# While loop rate
				#current_hz = 1/self.avg_cmd_vel[0]
				#time.sleep(max((current_hz-DESIRED_CTRL_HZ),0))

            # Update result and save model every 20 episodes
            if episode > 500 and episode % 20 == 0:
            	with open(os.path.join(self.results_path,'score'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
            		json.dump(self.score_list, outfile)

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

    def build_critic(self):
        goal_input = Input(shape=(2,))
        depth_image_input = Input(shape=(self.height, self.width, 1,))  
        actions_input = Input(shape=(self.action_size,))

        initializer = HeUniform()
        #depth_image_norm = BatchNormalization()(depth_image_input)
        #goal_norm = BatchNormalization()(goal_input)
        #actions_norm = BatchNormalization()(actions_input)

        c1 = Conv2D(32,3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(depth_image_input)
        c2 = Conv2D(32,3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c1)
        c2p = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(c2)
        c3 = Conv2D(64,3, strides=(2, 2), activation='relu', kernel_initializer = initializer)(c2p)
        c4 = Conv2D(64,3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c3)
        #c5 = Conv2D(128,3, strides=(1, 1), activation='relu', kernel_initializer = initializer)(c4)
        h0 = GlobalAveragePooling2D()(c4)

        fc1 = Dense(128, activation='relu', kernel_initializer = initializer)(h0)
        fc2 = Dense(64, activation='relu', kernel_initializer = initializer)(fc1)

        conc1 = concatenate([goal_input,fc2])
        h_state = Dense(128, activation='relu', kernel_initializer = initializer)(conc1)
        #h_action = Dense(64, activation='relu', kernel_initializer = initializer)(actions_input)
        conc2 = concatenate([h_state, actions_input])
        concat_h1 = Dense(128, activation='relu', kernel_initializer = initializer)(conc2)
        #concat_h1 = Dense(128, activation='relu', kernel_initializer = initializer)(concat_h1)

        output = Dense(1, activation='linear')(concat_h1)
        model = Model(inputs=[goal_input, depth_image_input, actions_input], outputs=[output], name = 'Critic')
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        model.summary()

        return model, adam
    
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
            goal = tf.reshape(state[0], [1,2])
            depth_image = tf.reshape(state[1], [1,self.height, self.width])
            pred_action = self.actor_model([goal, depth_image])
            #print("pred_action", pred_action)
            return [pred_action[0][0], pred_action[0][1]]

    def append_sample(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        #tf.print(len(self.memory))


    def train_model(self, target_train_start=False):
        time_check = time.time()
        mini_batch = random.sample(self.memory, self.batch_size)
       
        rewards=[]
        dones=[]
        for i in range(self.batch_size):
            if i == 0:
                tmp_goal = tf.expand_dims(mini_batch[i][0][0],0)
                goals = tmp_goal
                tmp_image= tf.expand_dims(mini_batch[i][0][1],0)
                images = tmp_image
                tmp_next_goal = tf.expand_dims(mini_batch[i][2][0],0)
                next_goals = tmp_next_goal
                tmp_next_image = tf.expand_dims(mini_batch[i][2][1],0)
                next_images = tmp_next_image
                tmp_action = tf.expand_dims(mini_batch[i][1],0)
                actions = tmp_action


            else:
                tmp_goal = tf.expand_dims(mini_batch[i][0][0],0)
                goals = tf.concat([goals, tmp_goal], axis=0) 
                tmp_image = tf.expand_dims(mini_batch[i][0][1],0)
                images = tf.concat([images, tmp_image], axis=0)      
                tmp_next_goal = tf.expand_dims(mini_batch[i][2][0],0)
                next_goals = tf.concat([next_goals, tmp_next_goal], axis=0)
                tmp_next_image = tf.expand_dims(mini_batch[i][2][1],0)
                next_images = tf.concat([next_images, tmp_next_image], axis=0)
                tmp_action = tf.expand_dims(mini_batch[i][1], 0)   
                actions = tf.concat([actions, tmp_action], axis=0)

            reward = np.asarray(mini_batch[i][3], dtype=np.float32)
            done = np.asarray(mini_batch[i][4], dtype=np.float32)
            rewards.append(reward)
            dones.append(done)
        rewards = np.array(rewards).reshape(self.batch_size,)
        dones = np.array(dones)

        #print('goals shape', goals.shape)
        #print('depth images shape', images.shape)
        #print('actions shape', actions.shape)
        #print('rewards shape', rewards.shape)
        #print('dones shape', dones.shape)

        #print('time to set batch', time.time()-time_check)
        #time_check = time.time()
        targets = self.train_critic(goals, images, actions, next_goals, next_images, rewards, dones, target_train_start)

        #if np.isnan(sum(critic_loss.numpy())):
        #    print("critic_loss ",critic_loss.np())
        #    raise ValueError("critic_loss is nan")
        #print('time to train critic', time.time()-time_check)

        #time_check = time.time()
        actor_loss = self.train_actor(goals,images)
        #if np.isnan(sum(actor_loss.numpy())):
        #    print("actor_loss ",actor_loss)
        #    raise ValueError("actor_loss is nan")
        #print('time to train actor', time.time()-time_check)

    #@tf.function
    def train_critic(self, goals, images, actions, next_goals, next_images, rewards, dones, target_train_start):
        #time_check = time.time()
        error = False

        try:
            if not target_train_start:
                target_actions = self.actor_model([next_goals, next_images])
                #target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])
                target_actions = np.asarray(target_actions, dtype= np.float32).reshape(self.batch_size, self.action_size)
            else:
                target_actions = self.target_actor_model([next_goals, next_images])
                #target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])
                target_actions = np.asarray(target_actions, dtype= np.float32).reshape(self.batch_size, self.action_size)

        except:
            #print("next_states ",next_states)
            #print("tensor next_states",next_states)
            error = True
        if error:
            print('Error in train critic, target action')
            target_actions = self.target_actor_model([next_goals, next_images])


        #print('time for target actions', time.time()-time_check)
        #print("target action shape", target_actions.shape)

        #time_check = time.time()
        if not target_train_start:
                target_q_values = self.critic_model([next_goals, next_images, target_actions])
                #target_q_values = tf.reshape(target_q_values, [self.batch_size,])
                target_q_values = np.asarray(target_q_values, dtype= np.float32).reshape(self.batch_size,)

        else:
                target_q_values = self.target_critic_model([next_goals, next_images, target_actions])
                #target_q_values = tf.reshape(target_q_values, [self.batch_size, ])
                target_q_values = np.asarray(target_q_values, dtype= np.float32).reshape(self.batch_size,)

        #print('time for target q values', time.time()-time_check)
        #time_check = time.time()
        #dones = tf.constant(dones)
        #rewards = tf.constant(rewards, shape=[self.batch_size,None])
        #print("rewards.shape ",rewards.shape)
        #print("rewards ",rewards)
        #print("target_q_values.shape ", target_q_values.shape)
        #print("dones.shape ", dones.shape)
        #ds = tf.math.subtract(tf.ones(shape=dones.shape),dones)
        #disc = tf.math.multiply(self.discount_factor,ds)
        #disc_targ= tf.math.multiply(target_q_values,disc)
        #targets = tf.math.add(rewards,disc_targ)
        targets = rewards + target_q_values*self.discount_factor*(np.ones(shape=dones.shape, dtype=np.float32)-dones)
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
        #print('time for targets', time.time()-time_check)
        #time_check = time.time()
        #inputs must be a list of tensors since we have a multiple inputs NN
        self.critic_model.train_on_batch([goals, images, actions], targets)
        #critic_loss = self.compute_critic_gradient(goals, images, actions, targets)
        #print('time for critic gradients', time.time()-time_check)
        return targets
    
    @tf.function
    def compute_critic_gradient(self, goals, images, actions, targets):
        with tf.GradientTape() as tape_critic:
            predicted_qs = self.critic_model([goals, images, actions])
            critic_loss = tf.reduce_mean(tf.math.square(targets - predicted_qs))
        critic_grad = tape_critic.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        return critic_loss

    @tf.function
    def train_actor(self, goals, images):
        with tf.GradientTape() as tape:
            a = self.actor_model([goals, images])
            tape.watch(a)
            q = self.critic_model([goals,images, a])
        dq_da = tape.gradient(q, a)
        #print('Action gradient dq_qa: ', dq_da)

        with tf.GradientTape() as tape:
            a = self.actor_model([goals,images])
        da_dtheta = tape.gradient(a, self.actor_model.trainable_variables, output_gradients= -dq_da)
        #print('Actor loss da_dtheta: ', da_dtheta)
        actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), da_dtheta))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))
        return q

def main(args=None):
    rclpy.init()
    pic4rl_training= Pic4rlTraining()

    pic4rl_training.get_logger().info('Node spinning ...')
    rclpy.spin(pic4rl_training)

    pic4rl_training.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
