import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from pic4rl_sensors.sensor import Sensor


class TestSensor(Sensor):
    def __init__(self,
                node_name = "TestSensor",
                msg_type = String,
                topic_name = "/chatter" ,
                qos_profile = 10
        ):
        super().__init__(node_name,String,topic_name,qos_profile)

def main(args=None):
    rclpy.init(args=args)

    test_sensor = TestSensor()

    rclpy.spin(test_sensor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    test_sensor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()