""" NOT A REAL TRAINING SCRIPT
Please check the README.md in located in this same folder
for an explanation of this script"""

import gym
import gym_gazebo2
import time

env = gym.make('TurtleBot3Lidar-v0', env_num=1)

#print("Resetting env")
env.reset()
time.sleep(5)
done = False

while not done:

    # take a random action
    rand_action = env.action_space.sample()
    observation, reward, done, info = env.step(rand_action)
    print("random action is: , ", rand_action)

env.close()
print(info)
