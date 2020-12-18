#!/usr/bin/env python3

import os

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from pic4rl_msgs.srv import State, Reset, Step

import json
import numpy as np
import random
import sys
import time
import math

#from pic4rl.pic4rl_environment import Pic4rlEnvironment
from pic4rl.agents.ddpg_visual_agent import DDPGVisualAgent

class Pic4Trainer():
	def __init__(self, agent, load_episode, episode_size, train_start, env):
		super().__init__()

		#self.env =  Pic4rlEnvironment()
		self.env = env()
		self.Agent = agent
		self.load_episode = load_episode
		self.episode_size = episode_size
		self.eval_episode = 20
		self.train_start = train_start
		self.train_score_list = []
		self.eval_score_list = []
		#self.results_path = '/home/mauromartini/mauro_ws/scores/lidar/last'
		self.results_path = '/home/results'

	def process(self):
		print("[Trainer.py] process")
		global_step = 0
		for episode in range(self.load_episode+1, self.episode_size):
			global_step += 1

			if global_step == self.train_start+1:
			    print('Start training models, global step:', global_step)

			score = self.make_episode(episode, global_step)
			print(
				"Episode:", episode,
				"score:", score,
				"memory length:", self.Agent.memory.mem_len,
				"epsilon:", self.Agent.epsilon)
				#"avg Hz:", 1/self.avg_cmd_vel[0])

			param_keys = ['epsilon']
			param_values = [self.Agent.epsilon]
			param_dictionary = dict(zip(param_keys, param_values))
			self.train_score_list.append(score)

			# Update result and save model every 20 episodes
			if episode > 600 and episode % 20 == 0:
				self.save_score(episode)
				self.Agent.save_model(episode, param_dictionary)

			# Epsilon (exploration policy)
			if self.Agent.epsilon > self.Agent.epsilon_min:
				self.Agent.epsilon *= self.Agent.epsilon_decay
			
			if episode % self.eval_episode == 0:
				score = self.make_episode(episode, training = False)
				print("Evaluation episode | Reward ", score)
				self.eval_score_list.append(score)

	def make_episode(self, episode, global_step = None, training = True):

		local_step = 0
		done = False
		score = 0

		# Reset environment
		state = self.env.reset(episode)

		while not done:
			local_step += 1
			#print('[trainer][make_episode] new local step at time: ', time.time())
			# Action based on the current state
			if local_step == 1:
				action = np.array([0.0, 0.0], dtype = np.float32)

			else:
				state = next_state
				action = self.Agent.get_action(state)
				#print('[trainer][make_episode] action taken')
				if np.any(np.isnan(action)):
					print("Action:", action)
					action = np.array([0.0, 0.0], dtype = np.float32)
				#print("Action:", action)
				#print("Action size:", action.shape)

			next_state, reward, done, info = self.env.step(action)

			# Save <s, a, r, s'> samples
			if local_step > 1 and training:
				self.Agent.remember(state, action, next_state, reward, done)

				if global_step > self.train_start:
					time_check = time.time()
					self.Agent.train()
					print('Total time for training:', time.time() - time_check)

					# UPDATE TARGET NETWORKS
					time_check= time.time()
					self.Agent.update_target_model_soft()
					print('time for target model update:', time.time()-time_check)
			
			score += reward
		return score

	def save_score(self, episode):
		with open(os.path.join(self.results_path,'train_score_episode'+str(episode)+'.json'), 'w') as outfile:
			json.dump(self.train_score_list, outfile)
		with open(os.path.join(self.results_path,'eval_score_episode'+str(episode)+'.json'), 'w') as outfile:
			json.dump(self.eval_score_list, outfile)

	def evalutate_Hz(self, init = False):
		if init:
				elf.start = time.time()
		else:
				end = time.time() 
				delta = end - self.start

				if  delta<=3:
						self.avg_cmd_vel[1]+=1
						self.avg_cmd_vel[0] = (self.avg_cmd_vel[0]*(self.avg_cmd_vel[1]-1)\
								 + delta)\
								/self.avg_cmd_vel[1]
				self.start = end


class Pic4VisualTrainer():
	def __init__(self, agent, load_episode, episode_size, train_start, env):

		self.env = env()
		self.Agent = agent
		self.load_episode = load_episode
		self.episode_size = episode_size
		self.train_start = train_start
		self.eval_episode = 20
		self.train_score_list = []
		self.eval_score_list = []
		self.results_path = '/home/mauromartini/mauro_ws/scores/camera/rosbot/last'

	def process(self):
		global_step = 0

		for episode in range(self.load_episode+1, self.episode_size):
			global_step += 1

			if global_step == self.train_start+1:
				print('Start training models, global step:', global_step)

			score = self.make_episode(episode, global_step)
			print(
				"Episode:", episode,
				"score:", score,
				"memory length:", self.Agent.memory.mem_len,
				"epsilon:", self.Agent.epsilon)
				#"avg Hz:", 1/self.avg_cmd_vel[0])

			param_keys = ['epsilon']
			param_values = [self.Agent.epsilon]
			param_dictionary = dict(zip(param_keys, param_values))
			self.train_score_list.append(score)

			if episode % self.eval_episode == 0:
				score = self.make_episode(episode, training = False)
				print("Evaluation episode | Reward ", score)
				self.eval_score_list.append(score)

			# Update result and save model every 20 episodes
			if episode > 600 and episode % 20 == 0:
				self.save_score(episode)
				self.Agent.save_model(episode, param_dictionary)

			# Epsilon (exploration policy)
			if self.Agent.epsilon > self.Agent.epsilon_min:
				self.Agent.epsilon *= self.Agent.epsilon_decay

	def make_episode(self, episode, global_step = None, training = True):
		local_step = 0
		done = False
		score = 0

		# Reset environment
		state = self.env.reset(episode)
		goal = state[0]
		depth_image = state[1]
		#print('goal info', goal)
		#print('depth image', depth_image)

		while not done:
			local_step += 1
			#print('new local step at time: ', time.time())
			# Action based on the current state
			if local_step == 1:
			    action = np.array([0.0, 0.0], dtype = np.float32)

			else:
				state = next_state
				goal = state[0]
				depth_image = state[1]

				action = self.Agent.get_action(goal, depth_image)

				if np.any(np.isnan(action)):
					print("Action:", action)
					action = np.array([0.0, 0.0], dtype = np.float32)
					#print("Action:", action)
					#print("Action size:", action.shape)

			next_state, reward, done, info = self.env.step(action)
			next_goal = next_state[0]
			next_image = next_state[1]

			# Save <s, a, r, s'> samples
			if local_step > 1 and training:
				self.Agent.remember(goal, depth_image, action, next_goal, next_image,  reward, done)

				if global_step > self.train_start:
					#time_check = time.time()
					self.Agent.train()
					#print('Total time for training:', time.time() - time_check)

					# UPDATE TARGET NETWORKS
					#time_check= time.time()
					self.Agent.update_target_model_soft()
					#print('time for target model update:', time.time()-time_check)
			score += reward
		return score

	def save_score(self, episode):
		with open(os.path.join(self.results_path,'train_score_episode'+str(episode)+'.json'), 'w') as outfile:
			json.dump(self.train_score_list, outfile)
		with open(os.path.join(self.results_path,'eval_score_episode'+str(episode)+'.json'), 'w') as outfile:
			json.dump(self.eval_score_list, outfile)
