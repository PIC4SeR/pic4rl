#!/usr/bin/env python3
#
# Authors: Enrico Sutera


# This node is responsible for intefacing with gazebo:
# Sensors reading
# Velocities communication
# Reset of enviroment
# Environment changes

import os
import random
import sys
import time

from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Empty
from pic4rl_msgs.srv import State, Reset
import numpy, math

#sensors
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

#from omnirob_sensors.Sensor import OdomSensor, LaserScanSensor
#from rclpy.executors import SingleThreadedExecutor



class Pic4rlGazebo(Node):
    def __init__(self):
        super().__init__('pic4rl_gazebo')  
        self.get_logger().info("Beginning initialization...")
        rclpy.logging.set_logger_level('pic4rl_gazebo', 10)
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            qos)
        
        # Initialise subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        # Initialise client
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
        self.reset_world_client = self.create_client(Empty, 'reset_world')
        self.pause_physics_client = self.create_client(Empty, 'pause_physics')
        self.unpause_physics_client = self.create_client(Empty, 'unpause_physics')

        # Initialise servers
        self.get_logger().info("Creating get_state service...")
        self.get_state_server = self.create_service(State, 'get_state', self.get_state_callback) 
        self.get_logger().info("Creating new_episode service...")
        self.new_episode_server = self.create_service(Reset, 'new_episode', self.new_episode_callback) 
        #self.get_logger().info("Creating send_twist service...")
        #self.send_twist = self.create_service(Empty, 'send_twist', self.pub_twist_callback)
        #rclpy.logging.initialize()
        #rclpy.logging.get_logger("init_finisd")

        self.get_logger().info("Init finished.")


    """##############################
    Sensors related callbacks
    ##############################"""

    def odom_callback(self, msg):
        #self.get_logger().debug("Receiving odom ...")
        self.odom = msg

    def odom_callback_tb3(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def euler_from_quaternion(self, quat):
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
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def scan_callback(self, msg):
        #self.get_logger().debug("Receiving scan ...")
        self.scan = msg

    """##############################
    Communication with environment callbacks
    ##############################"""

    def get_state_callback(self,request,response):
        self.get_logger().debug("get state callback request received...")
        response.data_received = True

        try:
            response.scan = self.scan
        except Exception as e:
            self.get_logger().debug("Scan data not available yet")
            response.data_received = False
        try:
            response.odom = self.odom
        except Exception as e:
            self.get_logger().debug("Odom data not available yet")  
            response.data_received = False
        
        self.get_logger().debug("sending state...")
        return response

    def new_episode_callback(self,request, response):
        self.get_logger().debug("Reset request received ...")

        self.get_logger().debug("Clearing variables ...")
        self.clear_variables()
        
        #self.get_logger().debug("Respawing robot ...")
        #self.respawn_robot()

        #self.get_logger().debug("Resetting simulation ...")
        #self.reset_simulation()
        
        self.get_logger().debug("Resetting world ...")
        self.reset_world()

        self.get_logger().debug("Respawing goal ...")
        self.respawn_entity(request.goal_pos_x, request.goal_pos_y)

        #self.get_logger().debug("Resetting variables ...")
        #self.reset_variables()

        self.get_logger().debug("Environment reset performed ...")
        response.success = True
        return response

    """##############################
    Auxiliar functions
    ##############################"""
    def clear_variables(self):
        try:
            del self.odom
        except:
            pass
        try:
            del self.scan
        except:
            pass

    def reset_variables(self):
        self.get_logger().debug("Un pausing for resetting variables ...")
        self.unpause()
        while True:
            try:
                #self.get_logger().debug("Checking variables ...")
                self.odom
                self.get_logger().debug("Odom is ok ...")
                self.scan
                self.get_logger().debug("Scan is ok ...")
                break
            except:
                rclpy.spin_once(self)
                self.get_logger().debug("Spinning once.")
                time.sleep(0.05)
        self.pause()


    """##############################
    Gazebo services
    ##############################"""

    # currently not used
    def respawn_robot(self):
        #to be fixed. It gives error when deleting the model for the eaerly  destrucion of nodes.s
        #print("pausing...")
        #self.pause()
        self.get_logger().debug("deleting entity...")
        self.delete_entity('omnirob')
        self.get_logger().debug("respawning entity...")
        entity_path='/home/enricosutera/ros_2_workspace/src/omnirob/omnirob_simulation/models/omnirob/model.sdf'
        initial_pose = Pose()
        #initial_pose.position.x = random.uniform(-3,3)
        #initial_pose.position.y = random.uniform(-3,3)
        self.spawn_entity(initial_pose,'omnirob',entity_path)
        #print("unpausing...")
        #self.unpause()
        #print("unpaused.")
        #time.sleep(1)

    def respawn_entity(self, goal_pose_x, goal_pose_y):

        #Goal initialization
        # Entity 'goal'
        self.entity_dir_path = os.path.dirname(os.path.realpath(__file__))
        #print(self.entity_dir_path)
        #self.entity_dir_path = self.entity_dir_path.replace(
        #    'omnirob_rl/omnirob_rl',
        #    'omnirob_simulation/models/goal_box')
        self.entity_dir_path = self.entity_dir_path.replace(
            'pic4rl/pic4rl/pic4rl',
            'pic4rl/pic4rl/models/goal_box')
        self.entity_path = os.path.join(self.entity_dir_path, 'model.sdf')
        self.entity = open(self.entity_path, 'r').read()
        self.entity_name = 'goal'

        self.get_logger().debug("deleting entity...")
        try:
            self.delete_entity('goal')
        except:
            pass
        self.get_logger().debug("respawning entity...")
        entity_path=self.entity_path
        initial_pose = Pose()
        initial_pose.position.x = goal_pose_x
        initial_pose.position.y = goal_pose_y
        self.spawn_entity(initial_pose,self.entity_name,entity_path)
        #print("unpausing...")
        #self.unpause()
        #print("unpaused.")
        #time.sleep(1)

    def reset_world(self):
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_world_client.call_async(req)    
        #time.sleep(1)

    # currently not used
    def reset_simulation(self):
        #self.pause()
        req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.reset_simulation_client.call_async(req)    
        #self.unpause()

    def delete_entity(self, entity_name):
        req = DeleteEntity.Request()
        #req.name = self.entity_name
        req.name = entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.delete_entity_client.call_async(req)
        self.get_logger().debug('Entity deleting request sent ...')

    def spawn_entity(self,pose = None, name = None, entity_path = None, entity = None):
        if not pose:
            pose = Pose()
        req = SpawnEntity.Request()
        req.name = name
        if entity_path:
            entity = open(entity_path, 'r').read()
        req.xml = entity
        req.initial_pose = pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.spawn_entity_client.call_async(req)

    # currently not used
    def pub_twist_callback(self, twist=Twist()):
        """self.get_logger().info("pub")
        self.unpause
        self.cmd_vel_pub.publish(twist)
        time.sleep(0.1)
        self.pause
        return
        """
        pass 

    def pause(self):
        req = Empty.Request()
        while not self.pause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.pause_physics_client.call_async(req) 

    def unpause(self):
        req = Empty.Request()
        while not self.unpause_physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.unpause_physics_client.call_async(req) 

def main(args=None):
    rclpy.init()
    pic4rl_gazebo = Pic4rlGazebo()
    #executor = SingleThreadedExecutor()
    #executor.add_node(omnirob_gazebo)
    #executor.add_node(omnirob_gazebo.lidar_node)
    #executor.add_node(omnirob_gazebo.odom_node)
    
    #executor.spin()
    pic4rl_gazebo.get_logger().info("Spinning ...")
    rclpy.spin(pic4rl_gazebo)

    pic4rl_gazebo.get_logger().info("Stop spinning ...")
    pic4rl_gazebo.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
