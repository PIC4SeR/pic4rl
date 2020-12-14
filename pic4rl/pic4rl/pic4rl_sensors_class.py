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
from pic4rl.pic4rl_sensors_inherited import GenericLaserScanSensor, OdometrySensor

class Sensors(GenericLaserScanSensor, OdometrySensor):
	def __init__(self,
				generic_laser_scan_sensor = True,
				odometry_sensor = True):

		if generic_laser_scan_sensor:
			GenericLaserScanSensor.__init__(self)

		if odometry_sensor:
			OdometrySensor.__init__(self)