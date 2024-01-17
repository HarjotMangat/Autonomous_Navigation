# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import tensorflow as tf

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()
        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.num_actions = Config.ACTION_SPACE
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences) - 1)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences

    def convert_data(self, experiences):
        #print("Printing values in experiences ", self.id)
        #for exp in experiences:
        #    print("Rewards: ",exp.reward)
        #    print("Actions: ", exp.action)
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))

        # wait for the prediction to come back
        p, v = self.wait_q.get()
        val = v.numpy().item()
        return p, val

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            prediction = np.asarray(prediction).astype('float64')   #normalizing prediction, was originally causing: ValueError: probabilities do not sum to 1
            prediction = prediction / np.sum(prediction)            #because of sum being equal to 1.000000000000xx or 0.9999999999999999xx
            action = np.random.choice(self.actions, p=prediction)
        return action

    def run_episode(self, env):
        env.reset()
        time.sleep(1.5)
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0
        step_iteration = 0

        while not done:
            # very first few frames
            if env.current_state is None:
                print("Agent ", self.id, " Empty state, filling frame buffer")
                env.step(None)  # NOOP, used to fill the first 4 frames of buffer
                continue

            prediction, value = self.predict(env.current_state)
            action = self.select_action(prediction)
            reward, done = env.step(action)
            print("Agent ",self.id, "Action was: ", action, "Reward was: ", reward)
            reward_sum += reward

            if Config.MAX_STEP_ITERATION < step_iteration:
                step_iteration = 0
                done = True

            exp = Experience(env.previous_state, action, prediction, reward, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value

                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            step_iteration += 1
            time_count += 1

    def run(self):
        global tf
        # randomly sleep up to 20 seconds. helps agents boot smoothly.
        time.sleep(np.random.rand()* 20)
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        env = Environment(self.id)
        print("waiting for env to intialize")
        time.sleep(10)

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode(env):
                total_reward += reward_sum
                total_length += len(r_) #+ 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))

        env.close()
