import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
#from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist #,Pose
from gazebo_msgs.msg import ModelStates
#import os

class TurtleBot3_Node(Node):

    observation_msg = None
    positiion_msg = None
    orientation_msg = None

    def __init__(self, env_id):
        super().__init__('Turtlebot3Node_'+ str(env_id))

        self.env_id = env_id

        # Publisher for sending actions to turtlebot3
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile=qos_profile_services_default)

        # Gazebo simulation related controls: Puase, Resume, and Reset
        self.pause = self.create_client(Empty, '/pause_physics')
        self.resume = self.create_client(Empty, '/unpause_physics')  
        #self.reset = self.create_client(Empty, '/reset_simulation')
        self.reset= self.create_client(Empty, '/reset_world')

        # Subscribe to the appropriate topics: sub_position for position and orientation; sub_scan for observation from laserscan
        # need to subscribe to a gazebo service for model_states to get waffle_depth position (x,y,z)
        # to help determine rewards in step() and get orientation (quaternion [x y z w])
        
        self._sub_position = self.create_subscription(ModelStates, '/gazebo/model_states', self.position_callback, qos_profile=qos_profile_sensor_data)
        self._sub_scan = self.create_subscription(LaserScan, '/scan', self.observation_callback, qos_profile=qos_profile_sensor_data)
    
    def observation_callback(self, message):
        """
        Callback method for the subscriber of '/scan' topic
        """
        self.observation_msg = message

    def position_callback(self, message):
        """
        Callback method for the subscriber of '/gazebo/model_states' topic
        """
        index = message.name.index('waffle_depth')
        pose = message.pose[index]
        self.positiion_msg = pose.position
        self.orientation_msg = pose.orientation          # <-- this orientation is in quaternarion form, need the yaw for degrees to goal
    
    def action_publish(self, action):

        if action == 0:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 1.25
            self.vel_pub.publish(vel_cmd)
        elif action == 1:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 1.0
            self.vel_pub.publish(vel_cmd)
        elif action == 2:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.5
            self.vel_pub.publish(vel_cmd)
        elif action == 3:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.6
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 4:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = -0.5
            self.vel_pub.publish(vel_cmd)
        elif action == 5:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = -1.0
            self.vel_pub.publish(vel_cmd)
        elif action == 6:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = -1.25
            self.vel_pub.publish(vel_cmd)
        elif action == None:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        else:
            raise AttributeError("Invalid Action: ", action)
        
        #print(self.env_id, " Action published: ", action)

    def pause_sim(self):

        #print(self.get_service_names_and_types())
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')
        pause_future = self.pause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, pause_future, timeout_sec=10)
        #if pause_future.result() is not None:
        #    print("got past service request: PAUSE")
        #else:
        #    print("!~~~~~~~~~~~~~~~~~~Failed to pause world~~~~~~~~~~~~~~~~~~!")

    def reset_sim(self):
        #print("Got to the reset_sim in Turtlebot3_Node")
        #print(self.get_service_names_and_types())
        while not self.reset.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')
        reset_future = self.reset.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, reset_future, timeout_sec=10)
        #if reset_future.result() is not None:
        #    print("got past service request: RESET")
        #else:
        #    print("!~~~~~~~~~~~~~~~~~~Failed to reset world~~~~~~~~~~~~~~~~~~!")

    def resume_sim(self):
        #print("Got to the resume sime function in Node")
        #print(self.get_service_names_and_types())
        while not self.resume.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/unpause_simulation service not available, waiting again...')
        resume_future = self.resume.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, resume_future, timeout_sec=10)
        #if resume_future.result() is not None:
        #    print("Got past the service request: RESUME")
        #else:
        #    print("!~~~~~~~~~~~~~~~~~~Failed to resume world~~~~~~~~~~~~~~~~~~!")
        

'''class Gazebo_Resume_Node(Node):
    def __init__(self):
        super().__init__('GazeboResume')
        # Gazebo simulation related controls: Puase, Resume, and Reset
        self.resume = self.create_client(Empty, '/unpause_physics')     

    def resume_sim(self):
        print("Got to the resume sim function in Node")
        #print(self.get_service_names_and_types())
        while not self.resume.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/unpause_simulation service not available, waiting again...')
        print('passed the while loop')
        resume_future = self.resume.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, resume_future, timeout_sec=10)
        print("Got past the service request")

class Gazebo_Pause_Node(Node):
    def __init__(self):
        super().__init__('GazeboPause')
        # Gazebo simulation related controls: Puase, Resume, and Reset
        self.pause = self.create_client(Empty, '/pause_physics')

    def pause_sim(self):
        print("Got to pause sim function in Node")
        #print(self.get_service_names_and_types())
        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')
        pause_future = self.pause.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, pause_future, timeout_sec=10)

class Gazebo_Reset_Node(Node):
    def __init__(self):
        super().__init__('GazeboReset')
        self.reset = self.create_client(Empty, '/reset_simulation')

    def reset_sim(self):
        print("Got to the reset_sim in Turtlebot3_Node")
        #print(self.get_service_names_and_types())
        while not self.reset.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('/reset_simulation service not available, waiting again...')
        print("got past while loop in reset_sim")
        reset_future = self.reset.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, reset_future, timeout_sec=10)
        #MultiThreadedExecutor.spin_until_future_complete(self, reset_future)
        print("got past service request")
        '''