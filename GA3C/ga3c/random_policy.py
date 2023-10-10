import gym
import gym_gazebo2
import time
from multiprocessing import Process
import os
from Config import Config
import numpy as np
import rclpy
from gym_gazebo2.envs.Turtlebot3.TB3node import TurtleBot3_Node

'''
env = gym.make('TurtleBot3Lidar-v0', env_num=1)

env.reset()
    
print("Successfully reset the environment")
time.sleep(5)
done = False
while not done:
    
    # take a random action
    rand_action = env.action_space.sample()
    observation, reward, done, info = env.step(rand_action)
    print("random action is: , ", rand_action)
    #print("observation is: ", observation)
    #time.sleep(.5)

env.reset()
time.sleep(5)
env.close()
print(info)
'''

class Experience:
    def __init__(self, state, action, reward, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done

class TestProcessAgent(Process):
    def __init__(self, id):
        super(TestProcessAgent, self).__init__()
        #if not rclpy.ok():
        #    rclpy.init()

        #self.turtlenode = TurtleBot3_Node(env_id=1)

        self.id = id
        self.exit_flag = 0
        #self.env = gym.make('TurtleBot3Lidar-v0', env_num=1)
        self.num_actions = 7
        self.actions = np.arange(self.num_actions)
        self.state = None

        self.discount_factor = Config.DISCOUNT
        # one frame at a time

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        #return experiences[:-1]
        return experiences

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        print("entered prediction function")


    def select_action(self, env):
        if self.id == 0:
            action = 3
        elif self.id == 1:
            action = 6
        else:
            action = 0
        #action = env.action_space.sample()
        return action

    def run_episode(self, env):
        print("Agent began running an episode")
        #time.sleep(1)
        #obs = self.env.reset()
        #self.turtlenode.reset_sim()
        self.state = env.reset()
        #time.sleep(2)
        #print("issue resetting env??")
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        #step_iteration = 0
        #print("value of done is: ", done)
        while not done:
            # very first few frames
            #current_state = self.node.observation_msg.ranges
            #print("current state of the env is :", env.take_observation())
            if self.state is None:
                print("empty state, taking random action")
                action = self.select_action(env)
                self.state, _, _,_ = env.step(action)
                continue

            action = self.select_action(env)
            next_state,reward, done, _ = env.step(action)
            time.sleep(0.5)
            reward_sum += reward
            exp = Experience(self.state, action, reward, done)
            experiences.append(exp)
            self.state = next_state

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 20 if done else -20

                updated_exps = TestProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    # We also override the run() method to define the target function for the process.
    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + 1 * 10))
        env = gym.make('TurtleBot3Lidar-v0', env_num=self.id, render=True)
        print("initializing env ", self.id)
        time.sleep(10)
        
        while self.exit_flag == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode(env):
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                #self.training_q.put((x_, r_, a_))
            #self.episode_log_q.put((datetime.now(), total_reward, total_length))
            print("total reward was: ", total_reward)

            self.exit_flag = 1

        print("Agent exited loop, exif_flag set to true?")
        env.close()

if __name__ == '__main__':
    p = TestProcessAgent(id=0)
    p1 = TestProcessAgent(id=1)
    #p2 = TestProcessAgent(id=2)
    p.start()
    time.sleep(10)
    p1.start()
    #time.sleep(15)
    #p2.start()
    p.join()
    p1.join()
    #p2.join()