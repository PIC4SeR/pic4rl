import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class Sensor(Node):

    def __init__(self,
                node_name,
                msg_type,
                topic_name,
                qos_profile = 10
                ):
        super().__init__(node_name)

        self.callback = self.sub_callback
        self.data = None

        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            self.sub_callback,
            qos_profile)
        #self.subscription  # prevent unused variable warning

        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Node started.')
        self.first_msg = False

    def timer_callback(self):
        pass
        #self.get_logger().info('Timer callback')
        #print(self.data)
        """msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1"""

    def sub_callback(self, msg):
        #self.get_logger().debug('Publishing: "%s"' % msg.data)
        self.data = msg
        if not self.first_msg :
                print(msg.data)
                self.first_msg = True




"""def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

"""
"""
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
    main()"""