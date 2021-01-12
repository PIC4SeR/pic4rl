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

from launch import LaunchDescription
from launch_ros.actions import Node

package_name = 'pic4rl'

def generate_launch_description():
    return LaunchDescription([
        Node(
            package= package_name,
            #node_namespace='pic4rl_training',
            node_executable='pic4rl_training',
            #node_name='pic4rl_training'
            node_name='pic4rl',
            output = 'screen'
        )
    ])

# def generate_launch_description():
#     return launch.LaunchDescription([
#         launch.actions.DeclareLaunchArgument(
#             'node_prefix',
#             default_value=[launch.substitutions.EnvironmentVariable('USER'), '_'],
#             description='Prefix for node names'),
#         launch_ros.actions.Node(
#             package='demo_nodes_cpp', node_executable='talker', output='screen',
#             node_name=[launch.substitutions.LaunchConfiguration('node_prefix'), 'talker']),
#     ])
