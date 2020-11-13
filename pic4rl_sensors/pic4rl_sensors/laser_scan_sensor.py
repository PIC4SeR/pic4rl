#import rclpy
#from rclpy.node import Node
from pic4rl_sensors.sensor import Sensor
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

class LaserScanSensor(Sensor)
    def __init__(self,
                node_name = "laser_scan_sensor",
                msg_type = LaserScan,
                topic_name = "/scan" ,
                qos_profile = qos_profile_sensor_data
        ):
        super().__init__(node_name,String,topic_name,qos_profile)
        
def main(args=None):
    rclpy.init(args=args)

    laser_scan_sensor = LaserScanSensor()

    rclpy.spin(laser_scan_sensor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    laser_scan_sensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()