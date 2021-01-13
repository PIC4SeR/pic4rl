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
from pic4rl.replay_buffer import ReplayBufferCamera
from pic4rl.NeuralNetworks import CriticCNNetwork, ActorCNNetwork

class DDPGVisualAgent:
	def __init__(self, state_size, image_height, image_width, action_size = 2, max_linear_vel = 0.8, max_angular_vel = 2, max_memory_size = 150000, load = False,
			gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05, tau = 0.01, batch_size = 64, noise_std_dev = 0.2):


		# State size and action size
		self.state_size = state_size 
		self.goal_shape = 2
		self.image_height = image_height
		self.image_width = image_width
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
		self.memory = ReplayBufferCamera(self.max_memory_size, self.goal_shape, self.image_height, self.image_width, self.action_size)

		self.actor = ActorCNNetwork(max_linear_velocity = self.max_linear_vel, max_angular_velocity = self.max_angular_vel, lr = 0.0001, name = 'actor')
		self.target_actor = ActorCNNetwork(max_linear_velocity = self.max_linear_vel, max_angular_velocity = self.max_angular_vel, lr = 0.0001, name = 'target_actor')
		self.critic = CriticCNNetwork(lr = 0.0005, name = 'critic')
		self.target_critic = CriticCNNetwork(lr = 0.0005, name = 'target_critic')

		self.actor.model().summary()
		self.critic.model().summary()
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
				'actor_weights_episode'+str(self.load_episode)+'.h5')

			target_actor_dir_path = os.path.join(
				self.target_actor.model_dir_path,
				'target_actor_weights_episode'+str(self.load_episode)+'.h5')

			critic_dir_path = os.path.join(
				self.critic.model_dir_path,
				'critic_weights_episode'+str(self.load_episode)+'.h5')

			target_critic_dir_path = os.path.join(
				self.target_critic.model_dir_path,
				'target_critic_weights_episode'+str(self.load_episode)+'.h5')

			self.actor.load_weights(actor_dir_path)
			self.critic.load_weights(critic_dir_path)
			self.target_actor.load_weights(target_actor_dir_path)
			self.target_critic.load_weights(target_critic_dir_path)
			with open(os.path.join(
					self.actor.model_dir_path,
					'epsilon_episode'+str(self.load_episode)+'.json')) as outfile:
				param = json.load(outfile)
				self.epsilon = param.get('epsilon')
   

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

	def get_action(self, goal, depth_image):
		if np.random.rand() <= self.epsilon:
			rnd_action = [random.random()*self.max_linear_vel, (random.random()*2-1)*self.max_angular_vel]
			#print("rnd_action",rnd_action)
			return rnd_action
		else:
			goal = tf.reshape(goal, [1,2])
			depth_image = tf.reshape(depth_image, [1,self.image_height, self.image_width,1])
			pred_action = self.actor(goal, depth_image)
			#noise = self.action_noise()
			#print("noise ", noise)
			#pred_action = pred_action.numpy() + noise
			#print("pred_action:", pred_action)
			return tf.reshape(pred_action, [2,])

	def remember(self, goal, depth_image, action, next_goal, next_image, reward, done):
		self.memory.store_transition(goal, depth_image, action, next_goal, next_image, reward, done)

	def train(self, target_train_start=False):
		if self.memory.mem_count < self.batch_size:
			return
		#time_check = time.time()
		#SET BATCH
		goals, images, actions, next_goals, next_images, rewards, dones = self.memory.sample_batch(self.batch_size)
		images = tf.expand_dims(images, axis=-1)
		next_images = tf.expand_dims(next_images, axis=-1)
		#print('goals shape', goals.shape)
		#print('actions shape', actions.shape)
		#print('images shape', images.shape)
		#states = tf.convert_to_tensor(states, dtype = tf.float32)
		#next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
		#print('tensor states shape', states.shape)
		#actions = tf.convert_to_tensor(actions, dtype = tf.float32)
		#print('tensor actions shape', actions.shape)
		rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
		dones = tf.constant(dones, dtype = tf.float32)
		#print(dones)
		#print('time to set batch', time.time()-time_check)
		#time_check = time.time()
		#targets = self.compute_q_targets(states, actions, next_states, rewards, dones)
		critic_loss = self.train_critic(goals, images, actions, next_goals, next_images, rewards, dones)
		if np.isnan(critic_loss.numpy()):
			rint("critic_loss ",critic_loss.np())
			raise ValueError("critic_loss is nan")       
		#print('time to train critic', time.time()-time_check)

		#time_check = time.time()
		actor_loss = self.train_actor(goals, images)
		if np.isnan(actor_loss.numpy()):
			print("actor_loss ",actor_loss)
			raise ValueError("actor_loss is nan")
		#print('time to train actor', time.time()-time_check)

	@tf.function
	def compute_td_error(self, goals, images, actions, next_goals, next_images, rewards, dones):
		rewards = tf.expand_dims(rewards, axis=1)
		dones = tf.expand_dims(dones, 1)
		rewards = tf.squeeze(rewards, axis=1)
		dones = tf.squeeze(dones, axis=1)

		not_dones = 1. - tf.cast(dones, dtype=tf.float32)
		next_act_target = self.target_actor(next_goals, next_images)
		next_q_target = self.target_critic(next_goals, next_images, next_act_target)
		target_q = rewards + not_dones * self.discount_factor * next_q_target
		current_q = self.critic(goals, images, actions)
		td_errors = tf.stop_gradient(target_q) - current_q
		return td_errors


	@tf.function
	def train_critic(self, goals, images, actions, next_goals, next_images, rewards, dones):
		with tf.GradientTape() as tape_critic:
			td_errors = self.compute_td_error(goals, images, actions, next_goals, next_images, rewards, dones)
			#print('td error shape', td_errors.shape)
			critic_loss = tf.reduce_mean((td_errors)**2)
			#critic_loss = mean_squared_error(target_q_values, predicted_qs)
			#print('critic loss ', critic_loss)
		critic_grad = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
		#print('critic grad', critic_grad)
		self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
		return critic_loss

	@tf.function
	def train_actor(self, goals, images):
		with tf.GradientTape() as tape:
			a = self.actor(goals, images)
			q = self.critic(goals, images, a)
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
