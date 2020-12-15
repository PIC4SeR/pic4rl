#!/usr/bin/env python3

import os
import tensorflow as tf
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
import numpy as np

class ActorNetwork(Model):
	def __init__(self, state_size, max_linear_velocity, max_angular_velocity, lr = 0.00025, 
			fc1_dims = 512, fc2_dims = 256, fc3_dims = 256, name = 'actor', **kwargs):
		super(ActorNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
			'/pic4rl/pic4rl/pic4rl',
			'/pic4rl/pic4rl/models/agent_lidar_model')

		#Velocity limits
		self.max_linear_velocity = max_linear_velocity
		self.max_angular_velocity = max_angular_velocity

		#Layers dimension
		self.state_size = state_size
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims

		#Learning rate and optimizer
		self.lr = lr
		self.optimizer = Adam(lr = self.lr)
		self.model_name = name

		#Layers definition
		#Input Layer
		self.state_input = Input(shape=(self.state_size,))
		

		#Hidden Layer
		self.fc1 = Dense(self.fc1_dims, activation='relu')
		self.fc2 = Dense(self.fc2_dims, activation='relu')
		self.fc3 = Dense(self.fc3_dims, activation='relu')
		#Output Layer
		self.linear_out = Dense(1, activation = 'sigmoid')
		self.angular_out = Dense(1, activation = 'tanh')

		with tf.device("/cpu:0"):
			self(tf.constant(np.zeros(shape=(1,) + (self.state_size,), dtype=np.float32)))

	def call(self, state):

		x = self.fc1(state)
		x = self.fc2(x)
		out = self.fc3(x)

		Linear_velocity = self.linear_out(out)*self.max_linear_velocity
		Angular_velocity = self.angular_out(out)*self.max_angular_velocity
		action = concatenate([Linear_velocity, Angular_velocity])

		return action

class CriticNetwork(Model):
	def __init__(self, state_size, action_size, lr = 0.001, fc_act_dims = 64,
		    fc1_dims = 256, fc2_dims = 256, fc3_dims = 128, name = 'critic', **kwargs):
		super(CriticNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
		    '/pic4rl/pic4rl/pic4rl',
		    '/pic4rl/pic4rl/models/agent_lidar_model')

		#Layers dimension
		self.state_size = state_size
		self.action_size = action_size
		self.fc1_dims = fc1_dims
		self.fc_act_dims = fc_act_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims

		#Learning rate, optimizer, loss, name
		self.lr = lr
		self.optimizer = Adam(lr = self.lr)
		self.loss = tf.keras.losses.MeanSquaredError()
		self.model_name = name

		#Layers definition
		#Input Layer
		self.state_input = Input(shape=(self.state_size,))
		self.action_input = Input(shape=(self.action_size,))

		#Hidden Layer
		self.fc1 = Dense(self.fc1_dims, activation='relu')
		self.fc_act = Dense(self.fc_act_dims, activation='relu')
		self.fc2 = Dense(self.fc2_dims, activation='relu')
		self.fc3 = Dense(self.fc3_dims, activation='relu')
		#Output Layer
		self.out = Dense(1, activation='linear')

		dummy_state = tf.constant(
			np.zeros(shape=(1,) + (self.state_size,), dtype=np.float32))
		dummy_action = tf.constant(
			np.zeros(shape=[1, self.action_size], dtype=np.float32))
		with tf.device("/cpu:0"):
			self(dummy_state, dummy_action)

	def call(self, state, action):

		x_state = self.fc1(state)
		x_action = self.fc_act(action)

		x_conc = concatenate([x_state, x_action])
		x = self.fc2(x_conc)
		x = self.fc3(x)

		q = self.out(x)

		return tf.squeeze(q, axis=1)

class ActorCNNetwork(Model):
	def __init__(self, max_linear_velocity, max_angular_velocity, height = 60, width  = 80,  lr = 0.00025, unitsc1 = 32, unitsc2 = 64,
			filt1_size = 3, filt2_size = 3, fc1_dims = 128, fc2_dims = 64, fc3_dims = 128,  name = 'actor', **kwargs):
		super(ActorCNNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
		    '/pic4rl/pic4rl/pic4rl',
		    '/pic4rl/pic4rl/models/agent_camera_model')

		#Velocity limits
		self.max_linear_velocity = max_linear_velocity
		self.max_angular_velocity = max_angular_velocity

		self.height = height
		self.width = width

		#Layers dimension
		self.filt1_size = filt1_size
		self.filt2_size = filt2_size
		self.conv1_dims = unitsc1
		self.conv2_dims = unitsc2

		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims

		#Learning rate and optimizer
		self.lr = lr
		self.optimizer = Adam(lr = self.lr)
		self.model_name = name

		#Layers definition
		#Input Layer
		#self.goal_input = Input(shape=(2,))
		#self.depth_image_input = Input(shape=(self.height, self.width, 1,))  
		self.image_shape = (1, height, width,1,)
		self.goal_shape = (2,)
		
		#Hidden Layer
		self.k_initializer = HeUniform()
		self.conv1 = Conv2D(self.conv1_dims, self.filt1_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		self.conv1_1 = Conv2D(self.conv1_dims, self.filt1_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		self.max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2))
		self.conv2 = Conv2D(self.conv2_dims, self.filt2_size, strides=(2, 2), activation='relu', kernel_initializer = self.k_initializer)
		self.conv3 = Conv2D(self.conv2_dims, self.filt2_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		self.fc1 = Dense(self.fc1_dims, activation='relu', kernel_initializer = self.k_initializer)
		self.fc2 = Dense(self.fc2_dims, activation='relu', kernel_initializer = self.k_initializer)
		self.fc3 = Dense(self.fc3_dims, activation='relu', kernel_initializer = self.k_initializer)
		#Output Layer
		self.linear_out = Dense(1, activation = 'sigmoid')
		self.angular_out = Dense(1, activation = 'tanh')
		
		# dummy_goal = tf.constant(
		# 	np.zeros(shape=(1,) + self.goal_shape, dtype=np.float32))
		# dummy_image = tf.constant(
		# 	np.zeros(shape= self.image_shape, dtype=np.float32))

		self.model()
		#self(dummy_goal, dummy_image)

	def call(self, goal, depth_image):
		c1 = self.conv1(depth_image)
		c1 = self.conv1_1(c1)
		cp = self.max_pool(c1)
		c2 = self.conv2(cp)
		c3 = self.conv3(c2)
		features = GlobalAveragePooling2D()(c3)

		xf = self.fc1(features)
		xf = self.fc2(xf)
		x = concatenate([xf, goal])

		out = self.fc3(x)

		Linear_velocity = self.linear_out(out)*self.max_linear_velocity
		Angular_velocity = self.angular_out(out)*self.max_angular_velocity
		action = concatenate([Linear_velocity, Angular_velocity])

		return action

	def model(self):
		goal_input = Input(shape=(2,))
		depth_image_input = Input(shape=(self.height, self.width, 1,))
		return Model(inputs = [goal_input, depth_image_input], outputs = self.call(goal_input, depth_image_input), name = self.model_name)

class CriticCNNetwork(Model):
	def __init__(self, height = 60, width = 80, lr = 0.001,  unitsc1 = 32, unitsc2 = 64, filt1_size = 3, filt2_size = 3,
		    fc1_dims = 128, fc2_dims = 64, fc3_dims = 128,  name = 'critic', **kwargs):
		super(CriticCNNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
		    '/pic4rl/pic4rl/pic4rl',
		    '/pic4rl/pic4rl/models/agent_camera_model')
		
		self.height = height
		self.width = width
		self.image_shape = (1, height, width,1,)
		self.goal_shape = (2,)
		
		#Layers dimension
		self.filt1_size = filt1_size
		self.filt2_size = filt2_size
		self.conv1_dims = unitsc1
		self.conv2_dims = unitsc2

		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims

		#Learning rate, optimizer, loss, name
		self.lr = lr
		self.optimizer = Adam(lr = self.lr)
		self.loss = tf.keras.losses.MeanSquaredError()
		self.model_name = name

		#Hidden Layer
		self.k_initializer = HeUniform()
		self.conv1 = Conv2D(self.conv1_dims, self.filt1_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		self.conv1_1 = Conv2D(self.conv1_dims, self.filt1_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		self.max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2,2))
		self.conv2 = Conv2D(self.conv2_dims, self.filt2_size, strides=(2, 2), activation='relu', kernel_initializer = self.k_initializer)
		self.conv3 = Conv2D(self.conv2_dims, self.filt2_size, strides=(1, 1), activation='relu', kernel_initializer = self.k_initializer)
		
		self.fc1 = Dense(128, activation='relu',  kernel_initializer = self.k_initializer)
		self.fc2 = Dense(64, activation='relu',  kernel_initializer = self.k_initializer)
		self.fc3 = Dense(128 , activation='relu', kernel_initializer = self.k_initializer)
		self.fc4 = Dense(128 , activation='relu', kernel_initializer = self.k_initializer)
		#Output Layer
		self.out = Dense(1, activation='linear')
		
		# dummy_goal = tf.constant(
		# 	np.zeros(shape = (1,) + self.goal_shape, dtype=np.float32))
		# dummy_image = tf.constant(
		# 	np.zeros(shape = self.image_shape, dtype=np.float32))
		# dummy_action = tf.constant(
		# 	np.zeros(shape = (1, 2,), dtype=np.float32))

		# #self(dummy_goal, dummy_image, dummy_action)
		self.model()

	def call(self, goal, depth_image, action):
		c1 = self.conv1(depth_image)
		c1 = self.conv1_1(c1)
		cp = self.max_pool(c1)
		c2 = self.conv2(cp)
		c3 = self.conv3(c2)
		features = GlobalAveragePooling2D()(c3)

		xf = self.fc1(features)
		xf = self.fc2(xf)
		x = concatenate([xf, goal])

		x = self.fc3(x)
		x_conc = concatenate([x, action])
		x2 = self.fc4(x_conc)

		q = self.out(x2)

		return tf.squeeze(q, axis=1)

	def model(self):
		goal_input = Input(shape=(2,))
		depth_image_input = Input(shape=(self.height, self.width, 1,))
		actions_input = Input(shape=(2,))
		return Model(inputs = [goal_input, depth_image_input, actions_input], outputs = self.call(goal_input, depth_image_input,actions_input), name = self.model_name)  
