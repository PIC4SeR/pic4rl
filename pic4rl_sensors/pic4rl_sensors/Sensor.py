activate_this_file = '/home/enricosutera/envs/tf2/bin/activate_this.py'

exec(compile(open(activate_this_file, "rb").read(), activate_this_file, 'exec'), dict(__file__=activate_this_file))



import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan,Image
from std_msgs.msg import String
import cv2
from cv_bridge import CvBridge

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

class RealSenseSensor(Sensor):
    def __init__(self,
                msg_type = Image,
                topic_name = '/intel_realsense_r200_depth/depth/image_raw',
                qos_profile = qos_profile_sensor_data
                ):
        super().__init__(msg_type,topic_name,qos_profile)
        self.name = "RealSense"
        self.bridge = CvBridge()    
        self.screen_height = 240
        self.screen_width = 320
        self.width = 224
        self.height = 224

    def process_data(self, **kwargs):

        screen_height = self.screen_height
        screen_width = self.screen_width
        if self.data is None:
            raise ValueError("No " + self.name + " data received")
        depth_image_raw = np.zeros((screen_height,screen_width), np.uint8)
        depth_image_raw = self.bridge.imgmsg_to_cv2(self.data, '32FC1')
        depth_image_raw = np.array(depth_image_raw, dtype= np.float32)
        #savetxt('/home/maurom/depth_images/text_depth_image_raw.csv', depth_image_raw, delimiter=',')
        #np.save('/home/maurom/depth_images/depth_image.npy', depth_image_raw)
        #cv2.imwrite('/home/maurom/depth_images/d_img_01.png', self.depth_image_raw)


        depth_image = self.depth_to_net_dim(depth_image_raw)
        #depth_image = np.array(depth_image, dtype= np.float32)
        #savetxt('/home/maurom/depth_images/text_depth_image.csv', depth_image, delimiter=',')
        #print('image shape: ', depth_image.shape)


        #check crop is performed correctly
        img = tf.convert_to_tensor(depth_image, dtype=tf.float32)
        img = tf.reshape(img, [screen_height,screen_width,1])
        width = self.width
        height = self.height
        img_crop = tf.image.crop_to_bounding_box(img, 2, 48, width,height)
        img_crop = tf.reshape(img_crop, [width,height])
        depth_image = np.asarray(img_crop, dtype= np.float32)
        #cv2.imwrite('/home/maurom/depth_images/d_img_crop.png', img_crop)
        self.image_size = depth_image.shape
        return {"depth_image":depth_image}


    def depth_to_net_dim(self,img):
        #Careful if the cutoff is in meters or millimeters!
        cutoff = 3.5
        img = self.depth_to_3ch(img, cutoff) # all values above 255 turned to white
        #cv2.imwrite('/home/maurom/depth_images/d_img_02.png', img) 
        img = self.depth_scaled_to_255(img) # correct scaling to be in [0,255) now
        #cv2.imwrite('/home/maurom/depth_images/d_img_03.png', img) 
        return img

    def depth_to_3ch(self,img, cutoff):
        #Useful to turn the background into black into the depth images.
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.flatten()
        img[img>cutoff] = 0.0 
        img = img.reshape([w,h])
        #for i in range(3):
        #    new_img[:,:,i] = img 
        return img

    def depth_scaled_to_255(self,img):
        assert np.max(img) > 0.0 
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        img = cv2.equalizeHist(img)
        #for i in range(3):
        #    img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img 

class TestStringSensor(Sensor):
    def __init__(self,
                msg_type = String,
                topic_name = 'topic',
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)