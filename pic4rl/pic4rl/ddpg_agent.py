#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

import json
import numpy as np
import random
import sys
import time
import math

from pic4rl.action_noise import OUActionNoise
from pic4rl.replay_buffer import ReplayBufferClassic
from pic4rl.NeuralNetworks import CriticNetwork, ActorNetwork

class DDPGLidarAgent:
	def __init__(self, state_size, action_size = 2, max_linear_vel = 0.8, max_angular_vel = 2, max_memory_size = 100000, load = False,
			gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05, tau = 0.01, batch_size = 64, noise_std_dev = 0.2):


		# State size and action size
		self.state_size = state_size 
		self.action_size = action_size 
		self.max_linear_vel = max_linear_vel
		self.max_angular_vel = max_angular_vel

		# DDPG hyperparameter
		self.tau = tau
		self.discount_factor = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.batch_size = batch_size

		# Replay memory
		self.max_memory_size = max_memory_size
		self.memory = ReplayBufferClassic(self.max_memory_size, self.state_size, self.action_size)

		# Build actor and critic models and target models
		# self.actor, self.actor_optimizer = self.build_actor()
		# self.critic, self.critic_optimizer = self.build_critic()
		# self.target_actor, _ = self.build_actor()
		# self.target_critic, _ = self.build_critic()
		self.actor = ActorNetwork(self.state_size, self.max_linear_vel, self.max_angular_vel, lr = 0.00025, fc1_dims = 256, fc2_dims = 128, fc3_dims = 128, name = 'actor')
		self.target_actor = ActorNetwork(self.state_size, self.max_linear_vel, self.max_angular_vel, lr = 0.00025, fc1_dims = 256, fc2_dims = 128, fc3_dims = 128, name = 'target_actor')
		self.critic = CriticNetwork(self.state_size, self.action_size, lr = 0.0005, fc_act_dims = 32,
			fc1_dims = 256, fc2_dims = 128, fc3_dims = 128, name = 'critic')
		self.target_critic = CriticNetwork(self.state_size, self.action_size, lr = 0.0005, fc_act_dims = 32,
			fc1_dims = 256, fc2_dims = 128, fc3_dims = 128, name = 'target_critic')

		self.update_target_model()

		#Exploration noise
		self.std_dev = noise_std_dev
		self.action_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

		#Load Models
		self.load = load
		self.load_episode = 0

		if self.load:
			actor_dir_path = os.path.join(
				self.actor.model_dir_path,
				'actor_stage1_episode'+str(self.load_episode)+'.h5')

			target_actor_dir_path = os.path.join(
				self.target_actor.model_dir_path,
				'actor_stage1_episode'+str(self.load_episode)+'.h5')

			critic_dir_path = os.path.join(
				self.critic.model_dir_path,
				'critic_stage1_episode'+str(self.load_episode)+'.h5')

			target_critic_dir_path = os.path.join(
				self.target_critic.model_dir_path,
				'critic_stage1_episode'+str(self.load_episode)+'.h5')

			self.actor.load_weights(actor_dir_path)
			self.critic.load_weights(critic_dir_path)
			self.target_actor.load_weights(target_actor_dir_path)
			self.target_critic.load_weights(target_critic_dir_path)
			# with open(os.path.join(
			# 		self.actor.model_dir_path,
			# 		#'epsilon_episode'+str(self.load_episode)+'.json')) as outfile:
			# 		'actor_stage1_episode600.json')) as outfile:
			# 	param = json.load(outfile)
			# 	self.epsilon = param.get('epsilon')


		# self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		# self.model_dir_path = self.model_dir_path.replace(
		#     '/pic4rl/pic4rl/pic4rl',
		#     '/pic4rl/pic4rl/models/agent_lidar_model')

		# self.stage = 1
		# self.actor_model_path = os.path.join(
		#     self.model_dir_path,
		#     'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')
		# self.critic_model_path = os.path.join(
		#     self.model_dir_path,
		#     'critic_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')

		# if self.load:
		#     self.actor.set_weights(load_model(self.actor_model_path).get_weights())

		#     with open(os.path.join(
		#             self.model_dir_path,
		#             'actor_stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
		#         param = json.load(outfile)
		#         self.epsilon = param.get('epsilon')

		#     self.critic.set_weights(load_model(self.critic_model_path).get_weights())   

		#     self.update_target_model()     

	def update_target_model(self):
	    self.target_actor.set_weights(self.actor.get_weights())
	    self.target_critic.set_weights(self.critic.get_weights())

	def update_target_model_soft(self):
		init_weights = self.actor.get_weights()
		update_weights = self.target_actor.get_weights()
		actor_weights = []
		for i in range(len(init_weights)):
			actor_weights.append(self.tau * init_weights[i] + (1 - self.tau) * update_weights[i])
		self.target_actor.set_weights(actor_weights)

		init_weights = self.critic.get_weights()
		update_weights = self.target_critic.get_weights()
		critic_weights = []
		for i in range(len(init_weights)):
			critic_weights.append(self.tau * init_weights[i] + (1 - self.tau) * update_weights[i])
		self.target_critic.set_weights(critic_weights)

	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			rnd_action = [random.random()*self.max_linear_vel, (random.random()*2-1)*self.max_angular_vel]
			#print("rnd_action",rnd_action)
			return rnd_action
		else:
			pred_action = self.actor(state.reshape(1, len(state)))
			
			#Generate and Add Noise 
			#noise = self.action_noise()
			#print("noise ", noise)
			#pred_action = pred_action.numpy() + noise
			#print("pred_action with noise", pred_action)
			pred_action = tf.reshape(pred_action, [2,])
			#print("pred_action", pred_action)
			#return [pred_action[0][0], pred_action[0][1]]
			return pred_action

	def remember(self, state, action, next_state, reward, done):
	    self.memory.store_transition(state, action, next_state, reward, done)

	def train(self, target_train_start=False):
		if self.memory.mem_count < self.batch_size:
			return
		#time_check = time.time()
		#SET BATCH
		states, actions, next_states, rewards, dones = self.memory.sample_batch(self.batch_size)
		#print('states shape', states.shape)
		#print('actions shape', actions.shape)
		states = tf.convert_to_tensor(states, dtype = tf.float32)
		next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
		#print('tensor states shape', states.shape)
		actions = tf.convert_to_tensor(actions, dtype = tf.float32)
		#print('tensor actions shape', actions.shape)
		rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
		dones = tf.constant(dones, dtype = tf.float32)
		#print(dones)
		#print('time to set batch', time.time()-time_check)
		#time_check = time.time()
		#targets = self.compute_q_targets(states, actions, next_states, rewards, dones)
		critic_loss = self.train_critic(states, actions, next_states, rewards, dones)
		if np.isnan(critic_loss.numpy()):
		    print("critic_loss ",critic_loss.np())
		    raise ValueError("critic_loss is nan")       
		#print('time to train critic', time.time()-time_check)
		#time_check = time.time()
		actor_loss = self.train_actor(states)
		if np.isnan(actor_loss.numpy()):
			print("actor_loss ",actor_loss)
			raise ValueError("actor_loss is nan")
		#print('time to train actor', time.time()-time_check)

	@tf.function
	def compute_td_error(self, states, actions, next_states, rewards, dones):
		rewards = tf.expand_dims(rewards, axis=1)
		dones = tf.expand_dims(dones, 1)
		rewards = tf.squeeze(rewards, axis=1)
		dones = tf.squeeze(dones, axis=1)

		not_dones = 1. - tf.cast(dones, dtype=tf.float32)
		next_act_target = self.target_actor(next_states)
		next_q_target = self.target_critic(next_states, next_act_target)
		target_q = rewards + not_dones * self.discount_factor * next_q_target
		current_q = self.critic(states, actions)
		td_errors = tf.stop_gradient(target_q) - current_q
		return td_errors


	@tf.function
	def train_critic(self, states, actions, next_states, rewards, dones):
		with tf.GradientTape() as tape_critic:
			td_errors = self.compute_td_error(states, actions, next_states, rewards, dones)
			#print('td error shape', td_errors.shape)
			critic_loss = tf.reduce_mean((td_errors)**2)
			#critic_loss = mean_squared_error(target_q_values, predicted_qs)
			#print('critic loss ', critic_loss)
		critic_grad = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
		#print('critic grad', critic_grad)
		self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
		return critic_loss

	@tf.function
	def train_actor(self, states):
		with tf.GradientTape() as tape:
			a = self.actor(states)
			q = self.critic(states, a)
			actor_loss = -tf.reduce_mean(q)
			#print('actor loss', actor_loss)
		actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
		#print('Action gradient v1: ', actor_grad)
		self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
		return actor_loss


	def save_model(self, episode, param_dictionary):
		actor_dir_path = os.path.join(
			self.actor.model_dir_path,
			'actor_weights_episode'+str(episode)+'.h5')
		self.actor.save_weights(actor_dir_path)

		target_actor_dir_path = os.path.join(
			self.target_actor.model_dir_path,
			'target_actor_weights_episode'+str(episode)+'.h5')
		self.target_actor.save_weights(target_actor_dir_path)

		critic_dir_path = os.path.join(
			self.critic.model_dir_path,
			'critic_weights_episode'+str(episode)+'.h5')
		self.critic.save_weights(critic_dir_path)

		target_critic_dir_path = os.path.join(
		 	self.target_critic.model_dir_path,
			'target_critic_weights_episode'+str(episode)+'.h5')
		self.target_critic.save_weights(target_critic_dir_path)

		with open(os.path.join(
			self.actor.model_dir_path,
				'epsilon_episode'+str(episode)+'.json'), 'w') as outfile:
			json.dump(param_dictionary, outfile)

#################
# OLD FUNCTIONS
#################
	def build_actor(self):
		state_input = Input(shape=(self.state_size,))
		h1 = Dense(512, activation='relu')(state_input)
		h2 = Dense(256, activation='relu')(h1)
		out1 = Dense(256, activation='relu')(h2)
		#out1 = Dropout(0.2)(out1)

		Linear_velocity = Dense(1, activation = 'sigmoid')(out1)*self.max_linear_vel
		Angular_velocity = Dense(1, activation = 'tanh')(out1)*self.max_angular_vel
		output = concatenate([Linear_velocity,Angular_velocity])

		model = Model(inputs=[state_input], outputs=[output])
		adam = Adam(lr= 0.00025)
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
		adam  = Adam(lr=0.0005)
		model.compile(loss="mse", optimizer=adam)
		model.summary()

		return model, adam

	def compute_q_targets(self, states, actions, next_states, rewards, dones):
		#time_check = time.time()
		target_actions = self.target_actor(next_states)  
		target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])
		target_q_values = self.target_critic(next_states, target_actions)
		target_q_values = tf.reshape(target_q_values, [self.batch_size, ])
		targets = rewards + target_q_values*self.discount_factor*(np.ones(shape=dones.shape, dtype=np.float32)-dones)
		#print("np targets shape ",targets.shape)
		#print("targets",targets)
		return targets

	@tf.function
	def train_critic2(self, states, actions, next_states, rewards, dones):
		with tf.GradientTape() as tape_critic:
			target_actions = self.target_actor(next_states)
			target_actions = tf.reshape(target_actions, [self.batch_size, self.action_size])

			target_critic_values = tf.squeeze(self.target_critic(next_states, target_actions), 1)
			target_q_values = rewards + target_critic_values*self.discount_factor*(np.ones(shape=dones.shape, dtype=np.float32)-dones)
			predicted_qs = tf.squeeze(self.critic(states, actions),1)
			critic_loss = mean_squared_error(target_q_values, predicted_qs)
		critic_grad = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
		#print('critic grad', critic_grad)
		self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

	@tf.function
	def train_actor2(self, states):
	# This is the precise implementation according to the DDPG paper,
	# however simplify the actor gradients expression is effective as well
		with tf.GradientTape() as tape:
			a = self.actor(states)
			#tape.watch(a)
			q = self.critic(states, a)
		dq_da = tape.gradient(q, a)
		#print('Action gradient dq_qa: ', dq_da)

		with tf.GradientTape() as tape:
			a = self.actor(states)
			theta = self.actor.trainable_variables
			#tape.watch(theta)
		da_dtheta = tape.gradient(a, theta, output_gradients= -dq_da)
		#print('Actor grad da_dtheta: ', da_dtheta)
		actor_gradients = list(map(lambda x: tf.divide(x, self.batch_size), da_dtheta))
		#print('actor grad', actor_gradients)
		self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
		#self.actor_optimizer.apply_gradients(zip(da_dtheta, self.actor_model.trainable_variables))