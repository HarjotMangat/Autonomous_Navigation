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

from threading import Thread
import numpy as np
import tensorflow as tf

from Config import Config


class ThreadTrainer(Thread):
    def __init__(self, server, id):
        super(ThreadTrainer, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.server = server
        self.exit_flag = False

    def run(self):
        while not self.exit_flag:
            batch_size = 0
            while batch_size <= Config.TRAINING_MIN_BATCH_SIZE:
                x_, r_, a_ = self.server.training_q.get()
                if batch_size == 0:
                    x__ = x_; r__ = r_; a__ = a_
                else:
                    print("+++++++++++GOT A BATCH LARGER THAN 0++++++++++++++")
                    x__ = np.concatenate((x__, x_))
                    r__ = np.concatenate((r__, r_))
                    a__ = np.concatenate((a__, a_))
                batch_size += x_.shape[0]
                print("batch size is: ", batch_size)
            
            if Config.TRAIN_MODELS:
                #trainX = tf.convert_to_tensor(x__, dtype=tf.float32)
                #trainR = tf.convert_to_tensor(r__, dtype=tf.float32)
                #trainA = tf.convert_to_tensor(a__, dtype=tf.float32)
                #tensorID = tf.convert_to_tensor(self.id, dtype=tf.int32)
                trainX = tf.Variable(x__, dtype=tf.float32)
                trainR = tf.Variable(r__, dtype=tf.float32)
                trainA = tf.Variable(a__, dtype=tf.float32)
                tensorID = tf.Variable(self.id, dtype=tf.int32)
                #self.server.train_model(x__, r__, a__, self.id)
                print("Sending training data to server")
                self.server.train_model(trainX, trainR, trainA, tensorID)
