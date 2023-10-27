import gym
gym.logger.set_level(40) # hide warnings
import time
import math
import numpy as np
import psutil
from scipy.stats import skew
from gym import utils, spaces
from gym_gazebo2.utils import ut_generic, ut_launch #, ut_mara, ut_math, ut_gazebo, tree_urdf, general_utils
from gym.utils import seeding
from gazebo_msgs.msg import ModelStates

#import copy

import os
#import signal
#import sys
#from gazebo_msgs.srv import SpawnEntity
#import subprocess
#import argparse
#import transforms3d as tf3d

# ROS 2
import rclpy
#from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from .TB3node import TurtleBot3_Node #, Gazebo_Pause_Node, Gazebo_Reset_Node, Gazebo_Resume_Node

#from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
#from sensor_msgs.msg import LaserScan
#from std_srvs.srv import Empty
#from geometry_msgs.msg import Twist,Pose

#from rclpy.qos import QoSProfile
#from rclpy.qos import QoSReliabilityPolicy
#from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing mara joint angles.
#from control_msgs.msg import JointTrajectoryControllerState
#from gazebo_msgs.srv import SetEntityState, DeleteEntity
#from gazebo_msgs.msg import ContactState, ModelState, GetModelList
#from std_msgs.msg import String
#from ros2pkg.api import get_prefix_path
#from builtin_interfaces.msg import Duration


class TurtleBot3Env(gym.Env):
    """
    Environment to interact with TurtleBot3 in Gazebo
    """

    def __init__(self, env_num, render):
        """
        Initialize the Turtlebot3 environemnt
        """
        # Manage command line args
        #args = ut_generic.getArgsParserMARA().parse_args()
        #self.gzclient = args.gzclient
        #self.multiInstance = args.multiInstance
        #self.port = args.port
        #print("Self.port is: ", self.port) #11345
        #self.gzclient = False
        self.gzclient = render
        self.multiInstance = True
        self.port = 11345

        # Create a dictionary to store the names of different worlds, their spawn points, and their corresponding target positions
        world = {}
        world['turtlebot3_world'] = {'spawn_point': [-1.5, -0.5], 'target_position': [2, 0]}
        world['turtlebot3_room'] = {'spawn_point': [-3.5, 3.75], 'target_position': [2.0, -2.5]}
        world['turtlebot3_house'] = {'spawn_point': [-6.5, 3.5], 'target_position': [6.5, 1]}
        #world['four_rooms'] = {'spawn_point': [-5.0, 5.0], 'target_position': [5.0, -4.0]}

        #if the env_num is 0 then we choose the first world, else if it is 1 then we choose the second world, etc. if it is 4 then we choose the first world again
        world_name = list(world.keys())[env_num % 3]
        #world_name = 'turtlebot3_house'

        # Pass this selected world to the launch file, so we can select world name and spawn point
        self.worldname = world_name
        print(env_num, " World name is: ", self.worldname + '.world')
        self.spawn_point = world[world_name]['spawn_point']

        # Launch turtlebot3 in a new Process
        self.launch_subp = ut_launch.startLaunchServiceProcess(
            ut_launch.generateLaunchDescriptionTurtlebot3(
                self.gzclient, self.multiInstance, self.port, self.worldname, self.spawn_point))

        #########################################################################
        # Create the node after the new ROS_DOMAIN_ID is set in generate_launch_description()
        if not rclpy.ok():
           rclpy.init()

        #rclypy.node --> class inherit from that. seperate turtlebot3evn from ros node stuff.
        #self.node = rclpy.create_node(self.__class__.__name__ + str(env_num))

        self.node = TurtleBot3_Node(env_id=env_num)
        #########################################################################

        #self.gzreset = Gazebo_Reset_Node()
        #self.gzpause = Gazebo_Pause_Node()
        #self.gzresume = Gazebo_Resume_Node()

        # class variables
        #self._observation_msg = None
        #self._position_msg = None

        self.id = env_num
        self.Position = None
        self.prevPosition = None
        self.Orientation = None
        self.prevOrientation = None
        self.max_episode_steps = 1024 #default value, can be updated from baselines
        self.iterator = 0
        self.currentDistance = None
        self.previousDistance = 1000

        #############################
        #   Environment hyperparams
        #############################
        # Target, where should the agent reach
        #self.targetPosition= np.asarray([2,0])
        self.targetPosition = np.asarray(world[world_name]['target_position'])

        #self.target_orientation = np.asarray([0., 0.7071068, 0.7071068, 0.]) # arrow looking down [w, x, y, z]
        #self.target_orientation = np.asarray([-0.4958324, 0.5041332, 0.5041331, -0.4958324]) # arrow looking opposite to MARA [w, x, y, z]

        #############################
        '''
        # Subscribe to the appropriate topics, taking into account the particular robot
        #self.vel_pub = self.node.create_publisher(Twist, '/cmd_vel', qos_profile=qos_profile_services_default)
        #self.pause = self.node.create_client(Empty, '/pause_physics')
        #self.resume = self.node.create_client(Empty, '/unpause_physics')
        #self._sub_scan = self.node.create_subscription(LaserScan, '/scan', self.observation_callback, qos_profile=qos_profile_sensor_data)
        #self._sub_scan = self.node.create_subscription(LaserScan, '/scan', self.observation_callback, qos_profile=qos_profile_services_default)
        #self._sub_coll = self.node.create_subscription(ContactState, '/gazebo_contacts', self.collision_callback, qos_profile=qos_profile_sensor_data)
        #self.reset_sim = self.node.create_client(Empty, '/reset_simulation')
        # Need to subscribe to a gazebo service for model_states to get waffle_depth position (x,y,z) to help determine rewards in step() and get orientation (quaternion [x y z w])
        #self._sub_position = self.node.create_subscription(ModelStates, '/gazebo/model_states', self.position_callback, qos_profile=qos_profile_sensor_data)
        '''

        #Esablish action space and observation space
        self.action_space = spaces.Discrete(7)

        high = np.inf*np.ones(3)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
        # Seed the environment
        self.seed()
        self.buffer_dist_rewards = []
        self.buffer_tot_rewards = []
        self.collided = 0

    '''def observation_callback(self, message):
        """
        Callback method for the subscriber of '/Scan' topic
        """
        self._observation_msg = message
    
    def position_callback(self, message):
        """
        Callback method for the subscriber of '/gazebo/model_states' topic
        """
        index = message.name.index('waffle_depth')
        pose = message.pose[index]
        position = pose.position

        #this orientation is in quaternarion form, need the yaw for degrees to goal
        orientation = pose.orientation

        self._position_msg = position
        self._orientation_msg = orientation
        #self._position_msg = message
    '''
    
    def set_episode_size(self, episode_size):
        self.max_episode_steps = episode_size

    def get_distance(self, position):
        # Measures Euler distance between robot(position) and goal(targetPosition)
        x = position.x - self.targetPosition[0]
        y = position.y - self.targetPosition[1]

        return math.sqrt(x*x + y*y)
    
    def get_orientation(self, orientation, position):

        # Get yaw of the turtlebot with respect to the gazebo world
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w

        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        robot_world_yaw = math.degrees(yaw)
        #print("Robot to world Yaw in degrees is: ", robot_world_yaw)
        
        # Get the angle from the robot to the goal
        robot_goal_calc = math.atan2(self.targetPosition[1] - position.y, self.targetPosition[0] - position.x)
        #print("Angle between robot and goal position in degrees is: ", math.degrees(robot_goal_calc))
        robot_goal_angle = math.degrees(robot_goal_calc)

        # Get the difference between the two angles (robot_world_yaw & robot_goal_angle) to determine how much the robot should rotate to face goal
        #diff = (robot_world_yaw - robot_goal_angle) % (math.pi * 2)
        #if diff >= math.pi:
        #    diff -= math.pi * 2

        diff = (robot_world_yaw - robot_goal_angle)
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
            
        #print("Difference between robot orientation and goal position is: ", diff)

        return diff

    def take_observation(self):
        """
        Take observation from the environment and return it as the current state.
        :return: state.
        """
        #Take an observation
        obs_message = None

        #while obs_message is None or int(str(self._observation_msg.header.stamp.sec)+(str(self._observation_msg.header.stamp.nanosec))) < self.ros_clock:
        
        while obs_message is None:
            rclpy.spin_once(self.node)
            #print(self.id, " waiting for new scan message")
            if len(self.node.observation_msg.ranges) == 270:
                obs_message = self.node.observation_msg
            else:
                obs_message = None
                print("ran into error with scan message, trying again")

        state = np.asarray(obs_message.ranges)
        #print("initial values in scan were: ", state)
        #print("Length of scan is: ", len(state))
        # Set the inf values to 5.0 and set any values <= 0.20 to 0.0
        for idx, item in enumerate(state):
            if item == float('inf'):
                state[idx] = 5.0
            elif item <= 0.20:
                state[idx] = 0.0
        #Normailze the values in the state to be between 0 and 1
        state = state / 5.0
        #print("Adjusted values in scan are: ", state)
        # Empty message after recieving it
        obs_message = None
        return state
    
    def take_position(self):
        """
        Take the position of the robot from the environment and return the position & orientation
        :return: position, orientation
        """
        #Initialize positionMsg to current position
        positionMsg = self.Position

        #Loop until we get new values for position & orientation
        while positionMsg == self.Position:
            rclpy.spin_once(self.node)
            #print(self.id, " waiting for position callback")

            positionMsg = self.node.positiion_msg
            orientationMsg = self.node.orientation_msg

        position = positionMsg
        orientation = orientationMsg
        return position, orientation

    def collision(self, obs):
        # Detect if there is a collision, return Boolean

        #print("Incoming ovbservation is: ", obs)
        if min(obs) == 0.0:
            print("++++++++++++++++++COLLISION DETECTED+++++++++++++++++++++++")

            self.collided += 1
            self.node.action_publish(None)
            return True
        else:
            return False
        
    def calculate_reward(self, collided, robot_to_goal_orientation, done):
        #Calculate Rewards. -20 for collision, 20 for reaching goal, small intermediate rewards for being closer to goal(by distance and orientation)
        
        #done = False
        reward = 0.0

        #calculating some value for intermediate rewards

        # distance between steps
        if self.prevPosition is None:
            dist_from_last_step = 0.0
        else:
            x = self.Position.x - self.prevPosition.x
            y = self.Position.y - self.prevPosition.y
            dist_from_last_step = math.sqrt(x*x + y*y) #distance from last step

        # distance of robot to end
        dist_to_end = self.currentDistance
        if self.previousDistance == 1000:
            dist_to_end_last = self.currentDistance
        else:
            dist_to_end_last = self.previousDistance

        # differnece in current distance and last distance
        if self.previousDistance == 1000:
            dist_to_end_diff = self.currentDistance
        else:
            dist_to_end_diff = abs(self.previousDistance - self.currentDistance) #distance to end of episode

        if self.prevOrientation is None:
            prev_robot_to_goal_orientation = 0
        else:
            prev_robot_to_goal_orientation = self.get_orientation(self.prevOrientation, self.prevPosition)

        rotations_cos_sum = math.cos(math.radians(robot_to_goal_orientation)) # [-1,1]

        #calculate the difference in rotations from last step to this step in the range of [0, pi]
        diff_rotations = math.fabs(math.fabs(math.radians(robot_to_goal_orientation)) - math.fabs(math.radians(prev_robot_to_goal_orientation))) # [0, pi]

        #start calculation of intermidiate rewards
        if dist_from_last_step != 0:
            dist_to_end_diff = dist_to_end_diff / dist_from_last_step # Normalize distance to end to [0, 1]
        else:
            dist_to_end_diff = 0

        if dist_to_end > math.sqrt(dist_from_last_step**2 + dist_to_end_last**2):
            dist_to_end_diff *= -6.0 # if distance to end is increasing, multiply by -6 to get negative reward [-6, 0]
        else:
            dist_to_end_diff *= 6.0
        
        if math.fabs(robot_to_goal_orientation) > math.fabs(prev_robot_to_goal_orientation):
            diff_rotations *= -3.0 # if rotation is increasing, multiply by -3 to get negative reward [-3*pi, 0]
        else:
            diff_rotations *= 2.0 # [0, 2*pi]

        reward += dist_to_end_diff #[-6, 6]
        reward += (3*rotations_cos_sum) #[-3, 3]
        reward += diff_rotations #[-3*pi, 2*pi]
        #print("reward at end of intermediate calculation is: ", reward)

        if collided: #Detected collision
            reward = -20
            done = True

        # check that robot reached goal
        elif self.currentDistance <= 0.25:
                reward = 20
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("++++++++++++++++++GOAL REACHED+++++++++++++++++++++++")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
                done = True

        return reward, done
    
    def encode_orientation(self, robot_to_goal_orientation):
        
        #one-hot encode the robot_to_goal_orientation (-180 to 180) in a vector of 128
        interval_size = 360/128 #detemines the size of each interval
        robot_to_goal_orientation = robot_to_goal_orientation + 180 #shifts the range from [-180, 180] to [0, 360]
        robot_to_goal_orientation = robot_to_goal_orientation // interval_size #divides the range into 128 intervals
        encoding = np.zeros(128) #creates a vector of 128 zeros
        encoding[int(robot_to_goal_orientation)] = 1 #sets the index of the interval to 1
        #print("The encoding of the orientation is: ", encoding)

        return encoding

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Implement the environment step abstraction. Execute action and returns:
            - action
            - observation
            - reward
            - done (status)
        """
        self.iterator+=1
        done = False

        #set initial position
        if self.Position is None:
            self.Position = self.node.positiion_msg

        # Resume simulation to take an action
        self.node.resume_sim()
        #self.gzresume.resume_sim()

        # Execute "action"
        self.node.action_publish(action)

        time.sleep(0.2)

        # Pause simulation
        #self.gzpause.pause_sim()
        self.node.pause_sim()

        #Take current position of robot
        self.Position, self.Orientation = self.take_position()
        # Take an observation
        obs = self.take_observation()
        #Calculate robot's distance and orientation to goal
        self.currentDistance = self.get_distance(self.Position)
        robot_to_goal_orientation = self.get_orientation(self.Orientation, self.Position)

        print(self.id, " Distance to goal is: ", self.currentDistance)
        print(self.id, " Orientation to goal is: ", robot_to_goal_orientation)
        
        # Calculate if the steps have been exausted
        steps_ended = bool(self.iterator == self.max_episode_steps)
        if steps_ended:
            done = True

        # Check for any collision. If we check the ranges of the laserscan and any values < 0.20 appear, we are too close to a wall/obstacle.
        collided = self.collision(obs)

        reward, done = self.calculate_reward(collided, robot_to_goal_orientation, done)

        print(self.id, " On step: ", self.iterator)

        encoding = self.encode_orientation(robot_to_goal_orientation)

        #concatenate the one-hot encoding to the observation
        obs = np.concatenate((obs, encoding), axis=None)
        #print("The new concatenated observation is: ",obs)
        #print("The shape of the observation is: ", obs.shape)

        self.previousDistance = self.currentDistance
        self.prevPosition = self.Position
        self.prevOrientation = self.Orientation

        self.buffer_tot_rewards.append(reward)
        info = {}
        if self.iterator % self.max_episode_steps == 0:

            max_tot_rew = max(self.buffer_tot_rewards)
            mean_tot_rew = np.mean(self.buffer_tot_rewards)
            std_tot_rew = np.std(self.buffer_tot_rewards)
            min_tot_rew = min(self.buffer_tot_rewards)
            skew_tot_rew = skew(self.buffer_tot_rewards)

            info = {"infos":{"ep_rew_max": max_tot_rew,"ep_rew_mean": mean_tot_rew,"ep_rew_min": min_tot_rew,\
                "ep_rew_std": std_tot_rew, "ep_rew_skew":skew_tot_rew}}
            self.buffer_tot_rewards = []
            self.collided = 0

        # Return the corresponding observations, rewards, etc.
        return obs, reward, done, info

    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        """
        self.iterator = 0
        # reset simulation
        #self.gzreset.reset_sim()
        self.node.reset_sim()

        print("+++++++++++++++++++ ", self.id ," Reset successfully++++++++++++++++++++")
        #self.ros_clock = rclpy.clock.Clock().now().nanoseconds

        self.node.resume_sim()
        time.sleep(1.5)
        self.node.pause_sim()

        #Take current position of robot
        self.Position, self.Orientation = self.take_position()
        robot_to_goal_orientation = self.get_orientation(self.Orientation, self.Position)

        # Take an observation
        obs = self.take_observation()

        encoding = self.encode_orientation(robot_to_goal_orientation)

        #concatenate the one-hot encoding to the observation
        obs = np.concatenate((obs, encoding), axis=None)

        # Return the corresponding observation
        return obs

    def close(self):
        print("Closing " + self.__class__.__name__ + " environment.")
        self.node.destroy_node()
        parent = psutil.Process(self.launch_subp.pid)
        for child in parent.children(recursive=True):
            child.kill()
        rclpy.shutdown()
        parent.kill()
