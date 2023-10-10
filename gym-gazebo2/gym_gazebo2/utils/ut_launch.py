import socket
import random
import os
import pathlib

from datetime import datetime
from billiard import Process

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchService, LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch.actions.execute_process import ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import gym_gazebo2
from gym_gazebo2.utils import ut_generic

def startLaunchServiceProcess(launchDesc):
    """Starts a Launch Service process. To be called from subclasses.

    Args:
         launchDesc : LaunchDescription obj.
    """
    # Create the LauchService and feed the LaunchDescription obj. to it.
    launchService = LaunchService()
    launchService.include_launch_description(launchDesc)
    process = Process(target=launchService.run)
    #The daemon process is terminated automatically before the main program exits,
    # to avoid leaving orphaned processes running
    process.daemon = True
    process.start()

    return process

def isPortInUse(port):
    """Checks if the given port is being used.

    Args:
        port(int): Port number.

    Returns:
        bool: True if the port is being used, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket1:
        return socket1.connect_ex(('localhost', port)) == 0

def getExclusiveNetworkParameters():
    """Creates appropriate values for ROS_DOMAIN_ID and GAZEBO_MASTER_URI.

    Returns:
        Dictionary {ros_domain_id (string), ros_domain_id (string)}
    """

    randomPortROS = random.randint(0, 230)
    randomPortGazebo = random.randint(10000, 15000)
    while isPortInUse(randomPortROS):
        print("Randomly selected port is already in use, retrying.")
        randomPortROS = random.randint(0, 230)

    while isPortInUse(randomPortGazebo):
        print("Randomly selected port is already in use, retrying.")
        randomPortGazebo = random.randint(10000, 15000)

    # Save network segmentation related information in a temporary folder.
    tempPath = '/tmp/gym-gazebo-2/running/'
    pathlib.Path(tempPath).mkdir(parents=True, exist_ok=True)

    # Remove old tmp files.
    ut_generic.cleanOldFiles(tempPath, ".log", 2)

    filename = datetime.now().strftime('running_since_%H_%M__%d_%m_%Y.log')

    file = open(tempPath + '/' + filename, 'w+')
    file.write(filename + '\nROS_DOMAIN_ID=' + str(randomPortROS) \
        + '\nGAZEBO_MASTER_URI=http://localhost:' + str(randomPortGazebo))
    file.close()

    return {'ros_domain_id':str(randomPortROS),
            'gazebo_master_uri':"http://localhost:" + str(randomPortGazebo)}

def generateLaunchDescriptionMara(gzclient, realSpeed, multiInstance, port, urdf):
    """
        Returns ROS2 LaunchDescription object.
        Args:
            realSpeed: bool   True if RTF must be set to 1, False if RTF must be set to maximum.
    """
    installDir = get_package_prefix('mara_gazebo_plugins')

    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = os.environ['GAZEBO_MODEL_PATH'] + ':' + installDir \
        + '/share'
    else:
        os.environ['GAZEBO_MODEL_PATH'] = installDir + "/share"

    if 'GAZEBO_PLUGIN_PATH' in os.environ:
        os.environ['GAZEBO_PLUGIN_PATH'] = os.environ['GAZEBO_PLUGIN_PATH'] + ':' + installDir \
        + '/lib'
    else:
        os.environ['GAZEBO_PLUGIN_PATH'] = installDir + '/lib'

    if port != 11345: # Default gazebo port
        os.environ["ROS_DOMAIN_ID"] = str(port)
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + str(port)
        print("******* Manual network segmentation *******")
        print("ROS_DOMAIN_ID=" + os.environ['ROS_DOMAIN_ID'])
        print("GAZEBO_MASTER_URI=" + os.environ['GAZEBO_MASTER_URI'])
        print("")
    elif multiInstance:
        # Exclusive network segmentation, which allows to launch multiple instances of ROS2+Gazebo
        networkParams = getExclusiveNetworkParameters()
        os.environ["ROS_DOMAIN_ID"] = networkParams.get('ros_domain_id')
        os.environ["GAZEBO_MASTER_URI"] = networkParams.get('gazebo_master_uri')
        print("******* Exclusive network segmentation *******")
        print("ROS_DOMAIN_ID=" + networkParams.get('ros_domain_id'))
        print("GAZEBO_MASTER_URI=" + networkParams.get('gazebo_master_uri'))
        print("")

    try:
        envs = {}
        for key in os.environ.__dict__["_data"]:
            key = key.decode("utf-8")
            if key.isupper():
                envs[key] = os.environ[key]
    except BaseException as exception:
        print("Error with Envs: " + str(exception))
        return None

    # Gazebo visual interfaze. GUI/no GUI options.
    if gzclient:
        gazeboCmd = "gazebo"
    else:
        gazeboCmd = "gzserver"

    # Creation of ROS2 LaunchDescription obj.

    if realSpeed:
        worldPath = os.path.join(os.path.dirname(gym_gazebo2.__file__), 'worlds',
                                 'empty.world')
    else:
        worldPath = os.path.join(os.path.dirname(gym_gazebo2.__file__), 'worlds',
                                 'empty_speed_up.world')

    launchDesc = LaunchDescription([
        ExecuteProcess(
            cmd=[gazeboCmd, '-s', 'libgazebo_ros_factory.so', '-s',
                 'libgazebo_ros_init.so', worldPath], output='screen', env=envs),
        Node(package='mara_utils_scripts', node_executable='spawn_mara.py',
             arguments=[urdf],
             output='screen'),
        Node(package='hros_cognition_mara_components',
             node_executable='hros_cognition_mara_components', output='screen',
             arguments=["-motors", installDir \
             + "/share/hros_cognition_mara_components/motors.yaml", "sim"]),
        Node(package='mara_contact_publisher', node_executable='mara_contact_publisher',
             output='screen')
    ])
    return launchDesc

def launchReal():
    #TODO: it is hard-coded
    os.environ["ROS_DOMAIN_ID"] = str(22)
    os.environ["RMW_IMPLEMENTATION"] = "rmw_opensplice_cpp"
    installDir = get_package_prefix('mara_gazebo_plugins')
    launchDesc = LaunchDescription([
        Node(package='hros_cognition_mara_components',
             node_executable='hros_cognition_mara_components',
             arguments=["-motors", installDir \
             + "/share/hros_cognition_mara_components/motors.yaml", "real"], output='screen')
    ])
    return launchDesc

def generateLaunchDescriptionTurtlebot3(gzclient, multiInstance, port, worldname, spawnpoint):
    """
        Returns ROS2 LaunchDescription object.
        Args:

    """
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    #x_pose = LaunchConfiguration('x_pose', default='-1.5')
    #y_pose = LaunchConfiguration('y_pose', default='-0.5')

    x_pose = LaunchConfiguration('x_pose', default=spawnpoint[0])
    y_pose = LaunchConfiguration('y_pose', default=spawnpoint[1])

    if worldname == 'turtlebot3_world':
        world = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'turtlebot3_world.world'
            #worldname + '.world'
        )
    elif worldname == 'turtlebot3_house':
        world = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'turtlebot3_house.world'
        )
    elif worldname == 'room':
        world = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'room.world'
        )
    elif worldname == 'four_rooms':
        world = os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'worlds',
            'four_rooms.world'
        )

    if port != 11345: # Default gazebo port
        os.environ["ROS_DOMAIN_ID"] = str(port)
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + str(port)
        print("******* Manual network segmentation *******")
        print("ROS_DOMAIN_ID=" + os.environ['ROS_DOMAIN_ID'])
        print("GAZEBO_MASTER_URI=" + os.environ['GAZEBO_MASTER_URI'])
        print("")
    elif multiInstance:
        # Exclusive network segmentation, which allows to launch multiple instances of ROS2+Gazebo
        networkParams = getExclusiveNetworkParameters()
        os.environ["ROS_DOMAIN_ID"] = networkParams.get('ros_domain_id')
        os.environ["GAZEBO_MASTER_URI"] = networkParams.get('gazebo_master_uri')
        print("******* Exclusive network segmentation *******")
        print("ROS_DOMAIN_ID=" + networkParams.get('ros_domain_id'))
        print("GAZEBO_MASTER_URI=" + networkParams.get('gazebo_master_uri'))
        print("")

    try:
        envs = {}
        for key in os.environ.__dict__["_data"]:
            key = key.decode("utf-8")
            if key.isupper():
                envs[key] = os.environ[key]
    except BaseException as exception:
        print("Error with Envs: " + str(exception))
        return None

    launchDesc = LaunchDescription()

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    launchDesc.add_action(gzserver_cmd)
    if gzclient:
        launchDesc.add_action(gzclient_cmd)
    launchDesc.add_action(robot_state_publisher_cmd)
    launchDesc.add_action(spawn_turtlebot_cmd)

    return launchDesc