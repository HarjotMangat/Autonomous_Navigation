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

import gym
from Config import Config
#from gym_gazebo2.envs.Turtlebot3.Turtlebot3_Env import TurtleBot3Env
import time
import gym_gazebo2


class GameManager:
    def __init__(self, game_name, display, id):
        self.game_name = game_name
        self.display = display

        print("Process ", id," made a new env")

        self.game_env = gym.make(game_name, env_num=id, render = Config.RENDER)
        #self.reset()

    def reset(self):
        observation = self.game_env.reset() 
        #print("waiting for env to reset")
        #time.sleep(5)
        return observation

    def step(self, action):
        #self._update_display()
        #print("entered game step")
        #print("selecting action ", action)
        observation, reward, done, info = self.game_env.step(action)
        return observation, reward, done, info

    #def _update_display(self):
    #    if self.display:
    #       self.game_env.render()

    def observation_size(self):
        #return self.game_env.observation_size()
        return Config.OBSERVATION_SIZE
    
    #def checkcollision(self):
    #    return self.game_env.collision((1,2,3,4))

    def close(self):
        self.game_env.close()
