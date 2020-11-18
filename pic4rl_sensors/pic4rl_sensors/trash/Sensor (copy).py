import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data



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