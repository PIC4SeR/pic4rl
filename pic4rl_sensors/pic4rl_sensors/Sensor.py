import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import numpy as np 

qos = QoSProfile(depth=10)

class Sensor():

    def __init__(self,
                msg_type,
                topic_name,
                qos_profile = 10
                ):
        self.msg_type = msg_type
        self.topic_name = topic_name
        self.callback = self.sub_callback
        self.qos = qos_profile
        self.name = NotImplementedError
        self.data = None

    def add_subscription(self):
        return self.msg_type, self.topic_name, self.callback ,self.qos

    def sub_callback(self, msg):
        #self.get_logger().debug('I heard: "%s"' % msg.data)
        #self.get_logger().debug('Topic ' + self.topic_name + ' msg received.')
        #print(msg)
        self.data = msg

    def process_data(self, **kwargs):

        raise NotImplementedError  

class OdomSensor(Sensor):
    def __init__(self,
                msg_type = Odometry,
                topic_name = 'odom',
                qos_profile = 10
                ):
        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)
        self.name = "Odometry"



    def process_data(self, **kwargs):
        if self.data is None:
            raise ValueError("No " + self.name + " data received")

        pos_x = self.data.pose.pose.position.x
        pos_y = self.data.pose.pose.position.y
        _,_,yaw = self.euler_from_quaternion(self.data.pose.pose.orientation)

        #return pos_x, pos_y, yaw
        return {"odom_pos_x":pos_x, 
                "odom_pos_y":pos_y,
                "odom_yaw": yaw}

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
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

class LaserScanSensor(Sensor):
    def __init__(self,
                msg_type = LaserScan,
                topic_name = 'scan',
                qos_profile = qos_profile_sensor_data
                ):
        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)
        self.name = "Lidar"

    def process_data(self, **kwargs):
        if self.data is None:
            raise ValueError("No " + self.name + " data received")
        laserscan_msg = self.data
        # There are some outliers (0 or nan values, they all are set to 0) that will not be passed to the DRL agent

        # Correct data:
        scan_range = []

        # Takes only sensed measurements
        for i in range(359):
            if laserscan_msg.ranges[i] == float('Inf'):
                scan_range.append(3.50)
            elif np.isnan(laserscan_msg.ranges[i]):
                scan_range.append(0.00)
            else:
                scan_range.append(laserscan_msg.ranges[i])

        """modified_scan_range = []

        for index in range(n_points):
            modified_scan_range.append(scan_range(int(index*modified_scan_range/n_points)))



        return modified_scan_range"""
        #return scan_range
        return {"scan_ranges":scan_range} 

class TestStringSensor(Sensor):
    def __init__(self,
                msg_type = String,
                topic_name = 'topic',
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)