import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


qos = QoSProfile(depth=10)

class Sensor():

    def __init__(self,
                msg_type,
                topic_name,
                qos_profile = 10
                ):
        self.msg_type = String
        self.topic_name = topic_name
        self.callback = self.sub_callback
        self.qos = qos_profile

        self.data = None

    def add_subscription(self):
        return self.msg_type, self.topic_name, self.callback ,self.qos

    def sub_callback(self, msg):
        #self.get_logger().debug('I heard: "%s"' % msg.data)
        #self.get_logger().debug('Topic ' + self.topic_name + ' msg received.')
        print(msg)
        self.data = msg

    def process_data(mode=None):

        raise NotImplementedError
        


class OdomSensor(Sensor):
    def __init__(self,
                msg_type = Odometry,
                topic_name = 'odom',
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)

class LaserScanSensor(Sensor):
    def __init__(self,
                msg_type = LaserScan,
                topic_name = 'scan',
                qos_profile = qos_profile_sensor_data
                ):

        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)

class TestStringSensor(Sensor):
    def __init__(self,
                msg_type = String,
                topic_name = 'topic',
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        super().__init__(msg_type,topic_name,qos_profile)



def main(args=None):
    """
    Run a Listener node standalone.
    This function is called directly when using an entrypoint. Entrypoints are configured in
    setup.py. This along with the script installation in setup.cfg allows a listener node to be run
    with the command `ros2 run examples_rclpy_executors listener`.
    :param args: Arguments passed in from the command line.
    """
    rclpy.init(args=args)
    try:
        sensor = Sensor()
        rclpy.spin(sensor)
    finally:
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    # Runs a listener node when this script is run directly (not through an entrypoint)
    main()