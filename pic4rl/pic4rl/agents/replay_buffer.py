#!/usr/bin/env python3

# General purpose
import time

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

class ReplayBufferCamera:
	def __init__(self, max_size, goal_shape, image_height, image_width, action_shape):
		self.mem_size = max_size
		self.mem_count = 0
		self.mem_len = 0

		self.goal_memory = np.zeros((self.mem_size, 2), dtype = np.float32)
		self.next_goal_memory = np.zeros((self.mem_size, 2), dtype = np.float32)
		self.image_memory = np.zeros((self.mem_size, image_height, image_width), dtype = np.float32)
		self.next_image_memory = np.zeros((self.mem_size, image_height, image_width), dtype = np.float32)
		self.action_memory = np.zeros((self.mem_size, action_shape), dtype = np.float32)
		self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
		self.done_memory = np.zeros(self.mem_size, dtype = np.bool)

	def store_transition(self, goal, image, action, next_goal, next_image, reward, done):
		index = self.mem_count % self.mem_size

		self.goal_memory[index] = goal
		self.next_goal_memory[index] = next_goal
		self.image_memory[index] = image
		self.next_image_memory[index] = next_image
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.done_memory[index] = done

		self.mem_count += 1
		self.mem_len = min(self.mem_count, self.mem_size)

	def sample_batch(self, batch_size):
		batch = np.random.choice(self.mem_len, batch_size, replace = False)

		goals = self.goal_memory[batch]
		next_goals = self.next_goal_memory[batch]
		images = self.image_memory[batch]
		next_images = self.next_image_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.done_memory[batch]

		return goals, images, actions, next_goals, next_images, rewards, dones

class ReplayBufferClassic:
	def __init__(self, max_size, state_shape, action_shape):
		self.mem_size = max_size
		self.mem_count = 0
		self.mem_len = 0

		self.state_memory = np.zeros((self.mem_size, state_shape), dtype = np.float32)
		self.next_state_memory = np.zeros((self.mem_size, state_shape), dtype = np.float32)
		self.action_memory = np.zeros((self.mem_size, action_shape), dtype = np.float32)
		self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
		self.done_memory = np.zeros(self.mem_size, dtype = np.bool)

	def store_transition(self, state, action, next_state, reward, done):
		index = self.mem_count % self.mem_size

		self.state_memory[index] = state
		self.next_state_memory[index] = next_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.done_memory[index] = done

		self.mem_count += 1
		self.mem_len = min(self.mem_count, self.mem_size)

	def sample_batch(self, batch_size):
		batch = np.random.choice(self.mem_len, batch_size, replace = False)

		states = self.state_memory[batch]
		next_states = self.next_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.done_memory[batch]

		return states, actions, next_states, rewards, dones