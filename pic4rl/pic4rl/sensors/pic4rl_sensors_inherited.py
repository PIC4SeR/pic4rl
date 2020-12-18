#!/usr/bin/env python3

# General purpose
import time
import numpy as np

# ROS related
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty

from geometry_msgs.msg import Twist

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

import collections


class GenericLaserScanSensor():
	def __init__(self):
		self.get_logger().info('/scan subscription')
		self.generic_laser_scan_sensor_sub = self.create_subscription(
			LaserScan,
			"/scan", 
			self.generic_laser_scan_cb,
			qos_profile_sensor_data
		)

	def generic_laser_scan_cb(self, msg):
		self.get_logger().debug('/scan Msg received')
		self.laser_scan_msg = msg

class OdometrySensor():
	def __init__(self):
		self.get_logger().info('/odom subscription')
		self.odometry_sensor_sub = self.create_subscription(
			Odometry,
			"/odom", 
			self.odometry_sensor_cb,
			10
		)

	def odometry_sensor_cb(self, msg):
		self.get_logger().debug('/odom Msg received')
		self.odometry_msg = msg

class GenericDepthCamera():
	def __init__(self):
		self.get_logger().info('/camera/depth/image_raw subscription')
		self.generic_depth_camera_sensor = self.create_subscription(
			Image,
			'/camera/depth/image_raw', 
			self.generic_depth_camera_cb,
			qos_profile_sensor_data
		)
		self.bridge = CvBridge()

	def generic_depth_camera_cb(self, msg):
		self.get_logger().debug('/camera/depth/image_raw Msg received')
		depth_image_raw = np.zeros((120,160), np.uint8)
		depth_image_raw = self.bridge.imgmsg_to_cv2(msg, '32FC1')
		
		self.generic_depth_camera_msg = msg
		self.generic_depth_camera_img = depth_image_raw