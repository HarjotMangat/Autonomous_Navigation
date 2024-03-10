import numpy as np
import tensorflow as tf
import os
import re

from Config import Config

'''@tf.function(input_signature=[tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, 8), dtype=tf.float32)])
def loss_function(y_true, y_pred):#, beta):
    action_index = y_true[:, :7] # splicing by 7 since num_actions is 7
    Y_r = tf.squeeze(y_true[:, 7:]) # splicing by 7 since num_actions is 7
    #tf.print("\naction_index(a): ", action_index, action_index.shape, "\nY_r(y_r): ", Y_r, Y_r.shape)
    
    policy_output = y_pred[:, :7] # splicing by 7 since num_actions is 7
    value_output = tf.squeeze(y_pred[:, 7:]) # splicing by 7 since num_actions is 7
    #tf.print("\npolicy_output(softmax_p): ", policy_output, policy_output.shape, "\nvalue_output(logits_v): ", value_output, value_output.shape)
        
    value_loss = 0.5 * tf.reduce_sum(tf.square(Y_r - value_output), axis=0)
    selected_action_prob = tf.reduce_sum(policy_output * action_index, axis=1)
    cost_p_1 = tf.math.log(tf.maximum(selected_action_prob, Config.LOG_EPSILON)) * (Y_r - tf.stop_gradient(value_output))
    cost_p_2 = -1 * Config.BETA_START * tf.reduce_sum(tf.math.log(tf.maximum(policy_output, Config.LOG_EPSILON)) * policy_output, axis=1)

    #tf.print("\nselected_action_prob: ", selected_action_prob, selected_action_prob.shape, "\ncost_p_1: ", cost_p_1, cost_p_1.shape, "\ncost_p_2: ", cost_p_2, cost_p_2.shape)
    
    #cost_p_1_agg = tf.reduce_sum(cost_p_1, axis=0)
    #cost_p_2_agg = tf.reduce_sum(cost_p_2, axis=0)
    
    cost_p_1_agg = tf.reduce_sum(cost_p_1)
    cost_p_2_agg = tf.reduce_sum(cost_p_2)
    cost_p = -(cost_p_1_agg + cost_p_2_agg)
    #tf.print("\ncost_p_1_agg: ", cost_p_1_agg, "\ncost_p_2_agg: ", cost_p_2_agg, "\ncost_p: ", cost_p, "\nvalue_loss(cost_v): ", value_loss, "\ntotal_loss: ", cost_p + value_loss)
    total_loss = cost_p + value_loss #total_loss:  15422.332

    return total_loss
    #return total_loss, value_loss, cost_p_1_agg, cost_p_2_agg, cost_p
    '''

class NetworkVPKeras:
    def __init__(self, device, model_name, num_actions):
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.observation_size = Config.OBSERVATION_SIZE
        self.rotation_size = Config.OBSERVATION_ROTATION_SIZE

        self.observation_channels = Config.STACKED_FRAMES

        self.learning_rate = tf.Variable(Config.LEARNING_RATE_START, dtype=tf.float32)
        self.beta = tf.Variable(Config.BETA_START,dtype=tf.float32)
        self.log_epsilon = Config.LOG_EPSILON
        if Config.TENSORBOARD:
            logdir = 'logs/' + self.model_name
            self.writer = tf.summary.create_file_writer(logdir)
        with tf.device(self.device):
            self.optimizer= tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, 
                                                        momentum=Config.RMSPROP_MOMENTUM,
                                                        epsilon=Config.RMSPROP_EPSILON, 
                                                        rho=Config.RMSPROP_DECAY)
            
            if Config.LOAD_CHECKPOINT:
                self.model = self.load()

            else:
                self.model = self._build_model()
                print(self.model.summary())
    
    
    def _build_model(self):
        self.inputs = tf.keras.Input(shape=(self.observation_size, self.observation_channels))
        self.Conv1 = tf.keras.layers.Conv1D(16, 9, strides=5, activation=tf.nn.relu, padding='same')(self.inputs)
        self.Conv2 = tf.keras.layers.Conv1D(32, 5, strides=3, activation=tf.nn.relu, padding='same')(self.Conv1)
        self.flattenlayer = tf.keras.layers.Flatten()(self.Conv2)
        self.Dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(self.flattenlayer)
        
        self.logits_p = tf.keras.layers.Dense(self.num_actions,activation=None,name="policy_output")(self.Dense1)
        self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.v = tf.keras.layers.Dense(1, activation=None, name="value_output")(self.Dense1)
        self.logits_v = tf.squeeze(self.v, axis=1)

        self.new_model = tf.keras.Model(inputs=self.inputs, outputs=[self.softmax_p, self.logits_v])

        return self.new_model 

    # Y_r is target value and a is action index
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, Config.OBSERVATION_SIZE, Config.STACKED_FRAMES), dtype=tf.float32),
                                   tf.TensorSpec(shape=(None), dtype=tf.float32),
                                   tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
                                   tf.TensorSpec(shape=(None), dtype=tf.int32)])
    def train(self, x, y_r, a, trainer_id):
        #y_true = tf.concat([a, y_r], axis=1)
        with tf.GradientTape() as tape:
            #forward pass
            logits_p, logits_v = self.model(x, training=True)
        
            #tf.print("\na: ", a, a.shape, "\ny_r: ", y_r, y_r.shape, "\ntrainer_id: ", trainer_id, trainer_id.shape)
            #tf.print("logits_p: ", logits_p, logits_p.shape, "\nlogits_v: ", logits_v, logits_v.shape)

            value_loss = 0.5 * tf.reduce_sum(tf.square(y_r - logits_v), axis=0)
            selected_action_prob = tf.reduce_sum(logits_p * a, axis=1)
            cost_p_1 = tf.math.log(tf.maximum(selected_action_prob, Config.LOG_EPSILON)) * (y_r - tf.stop_gradient(logits_v))
            cost_p_2 = -1 * self.beta * tf.reduce_sum(tf.math.log(tf.maximum(logits_p, Config.LOG_EPSILON)) * logits_p, axis=1)
            cost_p_1_agg = tf.reduce_sum(cost_p_1, axis=0)
            cost_p_2_agg = tf.reduce_sum(cost_p_2, axis=0)
            cost_p = -(cost_p_1_agg + cost_p_2_agg)
            loss_value = cost_p + value_loss
            #tf.print("\nselected_action_prob: ", selected_action_prob, selected_action_prob.shape, "\ncost_p_1: ", cost_p_1, cost_p_1.shape, "\ncost_p_2: ", cost_p_2, cost_p_2.shape)
            #tf.print("\ncost_p_1_agg: ", cost_p_1_agg, "\ncost_p_2_agg: ", cost_p_2_agg, "\ncost_p: ", cost_p, "\nvalue_loss(cost_v): ", value_loss, "\ntotal_loss: ", loss_value)

        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        #return loss_value, value_loss, cost_p1agg, cost_p2agg, costp

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, Config.OBSERVATION_SIZE, Config.STACKED_FRAMES), dtype=tf.float32)], reduce_retracing=True)
    def predict_p_and_v(self, x):
        p, v = self.model(x, training=False)
        return p, v
    
    def log(self, training_step, total_loss, value_loss, cost_p1agg, cost_p2agg, costp):
        with self.writer.as_default():
            #tf.summary.experimental.set_step(training_step)
            tf.summary.scalar('total_loss', total_loss, step=training_step)
            tf.summary.scalar('value_loss', value_loss, step=training_step)
            tf.summary.scalar('policy advantage_loss', cost_p1agg, step=training_step)
            tf.summary.scalar('policy entropy_loss', cost_p2agg, step=training_step)
            tf.summary.scalar('policy_loss', costp, step=training_step)
            self.writer.flush()

    def _checkpoint_filename(self, episode, policy_value):
        return 'checkpoints/%s_%08d_%04d.keras' % (self.model_name, episode, policy_value)
    
    def save(self, episode, policy_value):
        tf.keras.models.save_model(model=self.model, filepath=self._checkpoint_filename(episode=episode, policy_value=policy_value))

    def load(self):
        print("Loading model now")
        filename = ''

        if Config.LOAD_EPISODE == 0:
            episode = '00000000'
            for file in os.listdir('checkpoints'):
                if file.endswith('.keras'):
                    model, ep, val = file.split('_')
                    val, extension = val.split('.')
                    if ep > episode:
                        episode = ep
                        value = val
            filename = self._checkpoint_filename(episode=int(episode), policy_value=int(value))
        else:   #if Config.LOAD_EPISODE > 0
            for file in os.listdir('checkpoints'):
                if file.endswith('.keras'):
                    model, ep, val = file.split('_')        
                    val, extension = val.split('.')
                    if int(ep) == Config.LOAD_EPISODE:
                        filename = self._checkpoint_filename(episode=Config.LOAD_EPISODE, policy_value=int(val))
        print("filename is: ", filename)
        Config.LOAD_EPISODE = int(re.findall(r'\d+', filename)[0])
        Config.LOAD_POLICY_VALUE = float(re.findall(r'\d+', filename)[1])
        loaded_model = tf.keras.models.load_model(filepath=filename)

        return loaded_model