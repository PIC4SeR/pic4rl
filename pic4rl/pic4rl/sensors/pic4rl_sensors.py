#!/usr/bin/env python3

# General purpose
import time
import numpy as np
import random 
import math 

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

import collections

# LASERSCAN

def clean_laserscan(laserscan_data, laser_range = 3.5):
	# Takes only sensed measurements
	for i in range(359):
		if laserscan_data.ranges[i] == float('Inf'):
			laserscan_data.ranges[i] = laser_range #set range to max
		elif np.isnan(laserscan_data.ranges[i]):
			laserscan_data.ranges[i] = 0.0 #set range to 0
		else:
			pass # leave range as it is
	return laserscan_data

def laserscan_2_list(laserscan_data):
	laserscan_list = []
	for index in range(len(laserscan_data.ranges)):
		laserscan_list.append(laserscan_data.ranges[index])
	return laserscan_list #type: list (of float)

def laserscan_2_n_points_list(laserscan_data, n_points = 60):
	n_points_list = []
	len_laserscan_data = len(laserscan_data.ranges)
	for index in range(n_points):
		n_points_list.append(\
			laserscan_data.ranges[int(index*len_laserscan_data/n_points)]
			)
	return n_points_list #type: list (of float)

# ODOMETRY

def pose_2_xyyaw(odometry_data):
	# odometry_data:  nav_msgs/msg/Odometry
	pos_x = odometry_data.pose.pose.position.x
	pos_y = odometry_data.pose.pose.position.y
	_,_,yaw = euler_from_quaternion(odometry_data.pose.pose.orientation)
	return pos_x, pos_y, yaw #floats

def euler_from_quaternion(quat):
	"""
	Converts quaternion (w in last place) to euler roll, pitch, yaw
	quat = [x, y, z, w]
	"""
	x = quat.x
	y = quat.y
	z = quat.z
	w = quat.w

	sinr_cosp = 2 * (w*x + y*z)
	cosr_cosp = 1 - 2*(x*x + y*y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	sinp = 2 * (w*y - z*x)
	pitch = np.arcsin(sinp)

	siny_cosp = 2 * (w*z + x*y)
	cosy_cosp = 1 - 2 * (y*y + z*z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw

def goal_pose_to_distance(pos_x, pos_y, goal_pos_x, goal_pos_y):
	return math.sqrt((goal_pos_x-pos_x)**2
			+ (goal_pos_y-pos_y)**2)

def goal_pose_to_angle(pos_x, pos_y, yaw,goal_pos_x, goal_pos_y):
		path_theta = math.atan2(
			goal_pos_y-pos_y,
			goal_pos_y-pos_x)

		goal_angle = path_theta - yaw

		if goal_angle > math.pi:
			goal_angle -= 2 * math.pi

		elif goal_angle < -math.pi:
			goal_angle += 2 * math.pi
		return goal_angle

# CAMERA

def depth_rescale(img, cutoff):
	#Useful to turn the background into black into the depth images.
	w,h = img.shape
	#new_img = np.zeros([w,h,3])
	img = img.flatten()
	img[np.isnan(img)] = cutoff
	img[img>cutoff] = cutoff
	img = img.reshape([w,h])

	#assert np.max(img) > 0.0 
	#img = img/cutoff
	#img_visual = 255*(self.depth_image_raw/cutoff)
	img = np.array(img, dtype=np.float32)
	
	#img_visual = np.array(img_visual, dtype=np.uint8)
	#img_visual = cv2.equalizeHist(img_visual)
	#cv2.imwrite('/home/mauromartini/mauro_ws/depth_images/d_img_cutoff.png', img_visual)
    #for i in range(3):
    #    img[:,:,i] = cv2.equalizeHist(img[:,:,i])
	return img 

def clean_raw_depth(depth_image_msg):
	cutoff = 8

	img = np.array(depth_raw_img, dtype= np.float32)
	img = tf.reshape(img, [120,160,1])
	img_resize = tf.image.resize(img,[60,80])
	depth_image = tf.reshape(img_resize, [60,80])
	depth_image = np.array(depth_image, dtype= np.float32)
	depth_image = depth_rescale(depth_image, cutoff)
	return depth_image
