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


from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

# others

import collections


class GenericLaserScanSensor():
	def __init__(self):
		self.get_logger().info('/scan SUBCRIPTION STARTED')
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
		self.get_logger().info('/odom SUBCRIPTION STARTED')
		self.odometry_sensor_sub = self.create_subscription(
			Odometry,
			"/odom", 
			self.odometry_sensor_cb,
			10
		)

	def odometry_sensor_cb(self, msg):
		self.get_logger().debug('/odom Msg received')
		self.odometry_msg = msg
