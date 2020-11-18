import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_srvs.srv import Empty



#def main

class Sensor(Node):

    def __init__(self,
                node_name,
                msg_type,
                topic_name,
                qos_profile = 10,
                #service_msg_type = None,
                #service_name = None
                ):
        super().__init__(node_name)

        self.callback = self.sub_callback
        self.data = None

        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            self.sub_callback,
            qos_profile)

        #self.srv = self.create_service(service_msg_type, service_name+"_server", self.srv_callback)
        self.srv = self.create_service(Empty, topic_name+"_server", self.srv_callback)
        

        #if (service_msg_type != None and service_name != None):
        #    self.srv = self.create_service(service_msg_type, service_name, self.srv_callback)
        #    self.get_logger().info('Node service started.')
        #else:
        #    self.get_logger().warning('No service started.')
        #self.subscription  # prevent unused variable warning
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        #timer_period = 0.5  # seconds
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Node started.')
        self.first_msg = False

    def srv_callback(self, request, response):
        # Service callbacl action should be done here
        raise NotImplementedError  

    def process_data(self, **kwargs):
        # Main processing should be done here
        raise NotImplementedError  

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
                #print(msg.data)
                self.get_logger().info("First msg received: "+str(msg))
                self.first_msg = True