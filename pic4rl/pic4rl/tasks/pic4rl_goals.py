#!/usr/bin/env python3
#
# MIT License

# Copyright (c) 2021 PIC4SeR
# Authors: Enrico Sutera (enricosutera), Mauro Martini(maurom3197)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


# General purpose
import time
import numpy as np
import random 
import math 
import os
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Pose


class RandomGoal():
	"""
	Generate random goal coorditanes x,y given a range (min, max)
	and store them in 
		self.goal_pos_x
		self.goal_pos_y
	"""
	def __init__(self, goal_range = (-3,3)):
		self.get_logger().info('[RandomGoal] Initialization. ')
		self._goal_range = goal_range
		self.goal_pos_x = None
		self.goal_pos_y = None

		goal_path = os.path.join(get_package_share_directory('pic4rl'),\
					'gazebo/models/environment_elements', "goal_box")

		entity_path = os.path.join(goal_path, 'model.sdf')
		self.entity = open(entity_path, 'r').read()
		self.entity_name = 'goal'

		
	def new_goal(self):

		self.goal_pos_x = random.uniform(
			self._goal_range[0],self._goal_range[1])
		self.goal_pos_y = random.uniform(
			self._goal_range[0],self._goal_range[1])
		msg = '[RandomGoal] New goal x, y : {:.2f}, {:.2f}'.format(self.goal_pos_x, self.goal_pos_y)
		self.get_logger().info(msg)

		self.get_logger().debug("[RandomGoal] Deleting entity...")
		try:
			self.delete_entity(name = 'goal')
		except:
			pass
		self.get_logger().debug("respawning entity...")

		initial_pose = Pose()
		initial_pose.position.x = self.goal_pos_x
		initial_pose.position.y = self.goal_pos_y
		self.spawn_entity(pose = initial_pose,
							name = self.entity_name,
							entity = self.entity)

