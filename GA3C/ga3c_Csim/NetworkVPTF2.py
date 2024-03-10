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

import os
import re
import numpy as np
import tensorflow as tf
import sys

from Config import Config


class NetworkVP:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.observation_size = Config.OBSERVATION_SIZE
        self.rotation_size = Config.OBSERVATION_ROTATION_SIZE

        self.observation_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self._create_graph()

                self.sess = tf.compat.v1.Session(
                    graph=self.graph,
                    config=tf.compat.v1.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
                self.sess.run(tf.compat.v1.global_variables_initializer())

                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.compat.v1.global_variables()
                    self.saver = tf.compat.v1.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                

    def _create_graph(self):

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.observation_size, self.observation_channels], name='X')  #[batch_size, 1209, 4] state

        self.y_r = tf.compat.v1.placeholder(tf.float32, [None], name='Yr') #[batch_size] rewards

        self.var_beta = tf.compat.v1.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.compat.v1.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.n1 = self.conv1d_layer(self.x, 9, 16, 'conv11', stride=5)
        self.n2 = self.conv1d_layer(self.n1, 5, 32, 'conv12', stride=3)
        self.action_index = tf.compat.v1.placeholder(tf.float32, [None, self.num_actions]) #[batch_size, 7]
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1]) #logits_v: [-0.0372087769 -0.0396121666 -0.0325432718 ... -0.0342640653 -0.0271351878 -0.0392070264] TensorShape([None])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0) # cost_v: 14921.9014

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        '''logits_p: [[-0.047218103 0.0737353563 -6.12381846e-05 ... 0.0246349555 0.00245184638 0.0153825209]
                      [-0.0457983725 0.0738274083 0.00259433687 ... 0.025190901 0.00325815938 0.0142267346]
                      [-0.0455170237 0.0743581206 0.00386306457 ... 0.0233803056 0.0105269738 0.0140525587]
                      ...
                      [-0.0459367521 0.0712668598 -0.00399321318 ... 0.0253205 0.00408041105 0.0166395679]
                      [-0.0478867292 0.0734959245 0.00223142467 ... 0.0218092613 0.00998187624 0.0142297521]
                      [-0.0495245941 0.0670055 -0.00201696157 ... 0.0250985455 0.00649137236 0.0108124614]] TensorShape([None, 7]) '''
        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            '''softmax_p: [[0.134312779 0.15158169 0.140798271 ... 0.144318745 0.141152546 0.142989591]
                           [0.134415433 0.151496276 0.141080126 ... 0.144304335 0.141173795 0.142730802]
                           [0.134157658 0.151243448 0.140948668 ... 0.143726617 0.141891077 0.142392218]
                           ...
                           [0.134355813 0.151062727 0.140111014 ... 0.144278988 0.141246796 0.143031925]
                           [0.134163544 0.151478276 0.141058922 ... 0.143847749 0.142156437 0.142761603]
                           [0.134293944 0.150891498 0.140827909 ... 0.144698769 0.142031223 0.142646283]] TensorShape([None, 7])'''
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1) #selected_action_prob [0.134312779 0.144304335 0.151243448 ... 0.144278988 0.142156437 0.134293944] TensorShape([None])

            self.cost_p_1 = tf.math.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v)) #self.cost_p_1 [37.4965019 16.0278645 37.7158203 ... 7.98349524 8.69634056 40.0757675] TensorShape([None]) 
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.math.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1) #self.cost_p_2 [0.0194534175 0.0194535796 0.0194535181 ... 0.0194534529 0.019453587 0.0194538683] TensorShape([None]) 
        
        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0) #self.cost_p_1_agg 913.333496 
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0) #self.cost_p_2_agg 0.700330138
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg) #self.cost_p -914.033813 
        
        if Config.DUAL_RMSPROP:
            self.opt_p = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v #self.cost_all 14007.8672
            self.opt = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v) 
                                            for g,v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)
            
                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.compat.v1.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)
                #print_op = tf.print('\nlogits_v:', self.logits_v, self.logits_v.shape, '\ncost_v:', self.cost_v,
                #                    '\nlogits_p:', self.logits_p, self.logits_p.shape, '\nsoftmax_p:', self.softmax_p, self.softmax_p.shape, '\nselected_action_prob', self.selected_action_prob, self.selected_action_prob.shape,
                #                    '\nself.cost_p_1', self.cost_p_1, self.cost_p_1.shape, '\nself.cost_p_2', self.cost_p_2, self.cost_p_2.shape, 
                #                    '\nself.cost_p_1_agg', self.cost_p_1_agg, '\nself.cost_p_2_agg', self.cost_p_2_agg, '\nself.cost_p', self.cost_p, '\nself.cost_all', self.cost_all, '\n',
                #                    '\ncost_p:', self.cost_p, '\ncost_all:', self.cost_all, '\n', output_stream=sys.stdout)
                #self.grouped_op = tf.group(print_op, self.train_op)


    def _create_tensor_board(self):
        summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
        summaries.append(tf.compat.v1.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.compat.v1.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.compat.v1.summary.scalar("Ploss", self.cost_p))
        summaries.append(tf.compat.v1.summary.scalar("Vloss", self.cost_v))
        summaries.append(tf.compat.v1.summary.scalar("Totalloss", self.cost_all))
        #summaries.append(tf.compat.v1.summary.scalar("LearningRate", self.var_learning_rate))
        #summaries.append(tf.compat.v1.summary.scalar("Beta", self.var_beta))
        for var in tf.compat.v1.trainable_variables():
            summaries.append(tf.compat.v1.summary.histogram("weights_%s" % var.name, var))

        #summaries.append(tf.compat.v1.summary.histogram("activation_n1", self.n1))
        #summaries.append(tf.compat.v1.summary.histogram("activation_n2", self.n2))
        #summaries.append(tf.compat.v1.summary.histogram("activation_d2", self.d1))
        summaries.append(tf.compat.v1.summary.histogram("activation_v", self.logits_v))
        summaries.append(tf.compat.v1.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.compat.v1.summary.merge(summaries)
        self.log_writer = tf.compat.v1.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.compat.v1.variable_scope(name):
            w_init = tf.compat.v1.random_uniform_initializer(-d, d)
            b_init = tf.compat.v1.random_uniform_initializer(-d, d)
            w = tf.compat.v1.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.compat.v1.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.compat.v1.variable_scope(name):
            w_init = tf.compat.v1.random_uniform_initializer(-d, d)
            b_init = tf.compat.v1.random_uniform_initializer(-d, d)
            w = tf.compat.v1.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.compat.v1.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, filters=w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output
    
    def conv1d_layer(self, input, filter_size, out_dim, name, stride, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * in_dim)
        with tf.compat.v1.variable_scope(name):
            w_init = tf.compat.v1.random_uniform_initializer(-d, d)
            b_init = tf.compat.v1.random_uniform_initializer(-d, d)
            w = tf.compat.v1.get_variable('w',
                                shape=[filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.compat.v1.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv1d(input=input, filters=w, stride=stride, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def max_pool_2x2(self, x):
        return tf.nn.max_pool2d(input=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        return step

    def predict_single(self, x):
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
        return prediction

    def predict_p(self, x):
        prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
        return prediction
    
    def predict_p_and_v(self, x):
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x : x})
    
    def train(self, x, y_r, a, trainer_id):
        #tf.print('\nx:', x, x.shape, '\ny_r:', y_r, y_r.shape, '\na:', a, a.shape, '\n', output_stream=sys.stdout)
        '''x: array([[[0.04328614, 0.04345866, 0.04351369, 0.04348321],
        [0.04303023, 0.0430235 , 0.04314985, 0.04308167],
        [0.04288502, 0.0430827 , 0.0430518 , 0.04279165],
        ...,
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ]],

       [[0.12427622, 0.12399642, 0.4523057 , 0.11442588],
        [0.12363954, 0.12370695, 0.45263305, 0.11372504],
        [0.12306312, 0.12293066, 0.45235243, 0.11313476],
        ...,
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        ]]]) (35, 1209, 4)'''
        '''y_r: array([  8.18332819,   2.07401878,   0.5385212 , -20.        ,
                        11.18581149,   4.54043453,   1.8145397 , -20.        ,
                        46.40473523,  38.81595981,  30.59472151,  25.39755479,
                        18.44352678,  10.07771988,   5.86762691, -20.        ,
                        -42.64588601, -39.06705858, -35.51445959, -33.56929046,
                        -27.63955254, -24.82089753, -18.46199345, -16.16158797,
                        12.48744089,  -6.36337396,  -2.15485023,  33.43772567,
                        25.70119723,  16.45030962,   8.55226823, -20.        ,
                        14.3617673 ,   9.12138854, -20.        ]) (35,) '''
        '''a: array([[0., 0., 0., 0., 0., 1., 0.],
                    ...,
                    [0., 0., 0., 0., 0., 0., 1.]], dtype=float32) (35, 7)'''
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x : x, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)
        #self.sess.run(self.grouped_op, feed_dict=feed_dict)

        #step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        #self.log_writer.add_summary(summary,step)

    def log(self, x, y_r, a):
    
        feed_dict = self.__get_base_feed_dict()
        feed_dict.update({self.x : x, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)
    
    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)
       
    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))
