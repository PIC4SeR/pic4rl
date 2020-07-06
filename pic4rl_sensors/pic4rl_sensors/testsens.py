import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String

from pic4rl_sensors.Sensor import OdomSensor, TestStringSensor

rclpy.init()
ts = Node("test_sens")


tstring = TestStringSensor()
ts.create_subscription(*tstring.add_subscription())


#rclpy.init()
#ts = Node("test_sens")
#ts.odom = OdomSensor(node=ts)
#test = Sensorr()
#ts.create_subscription(String,'topic',test.sub_callback,qos)
#ts.create_subscription(*(test.add_subscription()))
#ts.create_subscription(test.add_subscription())
#ts.create_subscription(test.msg_type,
#                       test.topic_name,
#                       test.callback,
#                       test.qos)

"""       # Initialise subscribers
self.odom_sub = self.create_subscription(
Odometry,
'odom',
self.odom_callback,
qos)"""

try:
    rclpy.spin(ts)
finally:
    ts.destroy_node()
    rclpy.shutdown()