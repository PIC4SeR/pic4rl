import rclpy
#from rclpy.node import Node
from pic4rl_sensors.sensor import Sensor
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
#from std_msgs.msg import Empty
from std_srvs.srv import Empty

class LaserScanSensor(Sensor):
    def __init__(self):

        super().__init__(
                node_name = "laser_scan_sensor",
                msg_type = LaserScan,
                topic_name = "/scan" ,
                qos_profile = qos_profile_sensor_data,
                #service_msg_type = Empty,
                service_name = 'laser_scan_sensor_service'
                )
        
    def srv_callback(self, request, response):
        self.get_logger().info('Service called.')
        #raise NotImplementedError  


def main(args=None):
    rclpy.init(args=args)

    laser_scan_sensor = LaserScanSensor()

    rclpy.spin(laser_scan_sensor)

    laser_scan_sensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()