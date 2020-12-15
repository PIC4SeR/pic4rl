#!/usr/bin/env python3

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

import json
import numpy as np
import random
import sys
import time
import math

from pic4rl.pic4rl_environment import Pic4rlEnvironment
from pic4rl.ddpg_agent import DDPGLidarAgent
from pic4rl.ddpg_visual_agent import DDPGVisualAgent
from pic4rl.trainer import Pic4Trainer, Pic4VisualTrainer

from pic4rl.pic4rl_robots import MobileRobotState
from pic4rl.pic4rl_sensors import s7b3State

from pic4rl.pic4rl_sensors_class import Sensors
from pic4rl.pic4rl_env import Pic4rl


from gym import spaces

import gym

class Pic4rlRobot(Sensors, MobileRobotState, Pic4rl):
    def __init__(self):
        Pic4rl.__init__(self)
        Sensors.__init__(self, 
                        generic_laser_scan_sensor = True,
                        odometry_sensor = True)
        MobileRobotState.__init__(self)

        action =[
        [-0.5, 0.5], # x_speed 
        #[-0.5, 0.5], # y_speed
        [-1, 1] # theta_speed
        ]


        low_action = []
        high_action = []
        for i in range(len(action)):
            low_action.append(action[i][0])
            high_action.append(action[i][1])

        low_action = np.array(low_action, dtype=np.float32)
        high_action = np.array(high_action, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low_action,
            high=high_action,
            #shape=(1,),
            dtype=np.float32
        )
        
        """
        state
        """
        state =[
        [0., 5.], # goal_distance 
        [-math.pi, math.pi], # goal_angle
        #[-math.pi, math.pi] # yaw
        ]
        

        low_state = []
        high_state = []
        for i in range(len(state)):
            low_state.append(state[i][0])
            high_state.append(state[i][1])

        self.low_state = np.array(low_state, dtype=np.float32)
        self.high_state = np.array(high_state, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )


class Pic4rlTraining(Pic4rlEnvironment):
    def __init__(self):
        super().__init__()
        #rclpy.logging.set_logger_level('pic4rl_training', 20)
        #rclpy.logging.set_logger_level('pic4rl_environment', 10)

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        #self.env = Pic4rlEnvironment()
        #self.stage = int(stage)
	
        #self.avg_cmd_vel = [0.2,int(0)]
        #self.evalutate_Hz(init=True)

        # State size and action size
        self.state_size = 2 #goal distance, goal angle, lidar points
        self.action_size = 2 #linear velocity, angular velocity
        self.height = 60
        self.width = 80
        self.episode_size = 8000

        # Velocity limits and Controller freq
        self.max_linear_vel = 0.8
        self.max_angular_vel = 2
        self.control_freq = 5

        # Training parameters
        self.batch_size = 64
        self.train_start = 64
        #self.update_target_model_start = 128
        self.score_list = []

       # Load saved models
        self.load_model = False
        self.load_episode = 0


        #Instanciate DDPG Agent
        self.Agent = DDPGLidarAgent(state_size = self.state_size, action_size = self.action_size, 
             max_linear_vel = self.max_linear_vel, max_angular_vel= self.max_angular_vel, 
             max_memory_size = 200000, 
             load = self.load_model,
             gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05, 
             tau = 0.01, 
             batch_size = self.batch_size, 
             noise_std_dev = 0.2)  


        #self.Agent = DDPGVisualAgent(state_size = self.state_size, 
        #    image_height = self.height, image_width = self.width,
        #    action_size = self.action_size, 
        #    max_linear_vel = self.max_linear_vel, max_angular_vel= self.max_angular_vel, 
        #    max_memory_size = 150000, 
        #    load = self.load_model,
        #    gamma = 0.99, epsilon = 1.0, epsilon_decay = 0.998, epsilon_min = 0.05, 
        #    tau = 0.01, 
        #    batch_size = self.batch_size, 
        #    noise_std_dev = 0.2)  

        # Define and stat training process
        self.Trainer = Pic4Trainer(self.Agent, self.load_episode,\
                                    self.episode_size, self.train_start,\
                                    Pic4rlRobot)
        
        #self.Trainer = Pic4VisualTrainer(self.Agent, self.load_episode,\
        #                                 self.episode_size, self.train_start)
        self.Trainer.process()


def main(args=None):
    rclpy.init()
    pic4rl_training= Pic4rlTraining()

    pic4rl_training.get_logger().info('Node spinning ...')
    rclpy.spin(pic4rl_training)

    pic4rl_training.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()