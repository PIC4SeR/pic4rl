import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class Sensor():
    """
    A node with a single subscriber.
    This class creates a node which prints messages it receives on a topic. Creating a node by
    inheriting from Node is recommended because it allows it to be imported and used by
    other scripts.
    """

    def __init__(self,
                node_name,
                msg_type,
                topic_name,
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        #super().__init__(node_name)

        self.topic_name = topic_name

        self.sub = self.create_subscription(msg_type, topic_name, self.sub_callback, qos_profile)
        self.data = None

    def sub_callback(self, msg):
        #self.get_logger().debug('I heard: "%s"' % msg.data)
        self.get_logger().debug('Topic ' + self.topic_name + ' msg received.')
        self.data = msg

    def process_data(mode=None):

        raise NotImplementedError
        


class OdomSensor(Sensor):
    def __init__(self,
                node_name = 'odom_sensor',
                msg_type = Odometry,
                topic_name = 'odom',
                qos_profile = 10
                ):

        # Calls Node.__init__(node_name)
        super().__init__(node_name,msg_type,topic_name,qos_profile)

class LaserScanSensor(Sensor):
    def __init__(self,
                node_name = 'laserscan_sensor',
                msg_type = LaserScan,
                topic_name = 'scan',
                qos_profile = qos_profile_sensor_data
                ):

        # Calls Node.__init__(node_name)
        super().__init__(node_name,msg_type,topic_name,qos_profile)

class TestStringSensor(Sensor):
    def __init__(self,
                node_name = 'laserscan_sensor',
                msg_type = LaserScan,
                topic_name = 'scan',
                qos_profile = qos_profile_sensor_data
                ):

        # Calls Node.__init__(node_name)
        super().__init__(node_name,msg_type,topic_name,qos_profile)



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