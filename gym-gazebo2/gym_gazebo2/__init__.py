from gym.envs.registration import register

# Gazebo
# ----------------------------------------
# MARA
register(
    id='MARA-v0',
    entry_point='gym_gazebo2.envs.MARA:MARAEnv',
)

register(
    id='MARAReal-v0',
    entry_point='gym_gazebo2.envs.MARA:MARARealEnv',
)

register(
    id='MARACamera-v0',
    entry_point='gym_gazebo2.envs.MARA:MARACameraEnv',
)

register(
    id='MARAOrient-v0',
    entry_point='gym_gazebo2.envs.MARA:MARAOrientEnv',
)

register(
    id='MARACollision-v0',
    entry_point='gym_gazebo2.envs.MARA:MARACollisionEnv',
)

register(
    id='MARACollisionOrient-v0',
    entry_point='gym_gazebo2.envs.MARA:MARACollisionOrientEnv',
)

register(
    id='MARARandomTarget-v0',
    entry_point='gym_gazebo2.envs.MARA:MARARandomTargetEnv',
)

register(
    id='TurtleBot3Lidar-v0',
    entry_point='gym_gazebo2.envs.Turtlebot3:TurtleBot3Env',
)
