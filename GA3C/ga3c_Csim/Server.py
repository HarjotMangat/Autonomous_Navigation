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

from multiprocessing import Queue
import tensorflow as tf
import time

from Config import Config
from Environment import Environment
from NetworkVPTF2 import NetworkVP
from NetworkKeras import NetworkVPKeras #Keras model import
from ProcessAgent import ProcessAgent
from ProcessStats import ProcessStats
from ThreadDynamicAdjustment import ThreadDynamicAdjustment
from ThreadPredictor import ThreadPredictor
from ThreadTrainer import ThreadTrainer

from threading import Lock


class Server:
    def __init__(self):
        self.stats = ProcessStats()

        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)

        #self.model = NetworkVP(Config.DEVICE, Config.NETWORK_NAME, Environment().get_num_actions())
        self.model = NetworkVPKeras(Config.DEVICE, Config.NETWORK_NAME, Environment().get_num_actions()) #keras model
        if Config.LOAD_CHECKPOINT:
            #self.stats.episode_count.value = self.model.load()
            self.stats.episode_count.value = Config.LOAD_EPISODE #Keras loading
            self.stats.best_policy_value.value = Config.LOAD_POLICY_VALUE

        self.training_step = 0
        self.frame_counter = 0

        self.model_lock = Lock()

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

    def add_agent(self):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def train_model(self, x_, r_, a_, trainer_id):
        with self.model_lock:
            trainX = tf.convert_to_tensor(x_, dtype=tf.float32)
            trainR = tf.convert_to_tensor(r_, dtype=tf.float32)
            trainA = tf.convert_to_tensor(a_, dtype=tf.float32)
            tensorID = tf.convert_to_tensor(trainer_id, dtype=tf.int32)
            #trainR = tf.reshape(trainR, [-1, 1])
            self.model.train(trainX, trainR, trainA, tensorID)
        #self.model.train(x_, r_, a_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            pass
            #self.model.log(x_, r_, a_)
            #self.model.log(self.stats.training_count.value, tot, val, cp1agg, cp2agg, pol)


    def predict(self, state):
        with self.model_lock:
            state_ = tf.convert_to_tensor(state, dtype=tf.float32)
            p,v = self.model.predict_p_and_v(state_)
            p = p.numpy().tolist()
            v = v.numpy().tolist()
        return p, v

    def save_model(self):
        #pass
        #self.model.save(self.stats.episode_count.value)
        self.model.save(episode=self.stats.episode_count.value, policy_value=self.stats.policy_value.value) #Keras save model

    def main(self):
        self.stats.start()
        self.dynamic_adjustment.start()

        if Config.PLAY_MODE:
            for trainer in self.trainers:
                trainer.enabled = False

        learning_rate_multiplier = (
                                       Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES and not self.stats.end_training_value.value:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()