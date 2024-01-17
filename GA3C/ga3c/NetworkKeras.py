import numpy as np
import tensorflow as tf
import os
import re

from Config import Config

#@tf.keras.utils.register_keras_serializable(package='NetworkVPKeras', name="loss_function1")
#@tf.function
def loss_function(y_true, y_pred, beta):
    '''#component of the loss function for value function
    self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)
        
    self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
    self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

    #component of the loss function for the policy
    self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                * (self.y_r - tf.stop_gradient(self.logits_v))
    self.cost_p_2 = -1 * self.var_beta * \
                tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                            self.softmax_p, axis=1)
      
    #aggregating components of the policy loss function
    self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
    self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
    self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

    #computing a total loss for the model, using policy loss and value function loss
    self.cost_all = self.cost_p + self.cost_v'''
 
    action_index, Y_r = y_true #action_index is a tensor of shape (batch_size, num_actions), Y_r is a tensor of shape (batch_size, 1)
    policy_output, value_output = y_pred # policy_output is a tensor of shape (batch_size, num_actions), value_output is a tensor of shape (batch_size, 1)
 
    value_loss = 0.5 * tf.reduce_sum(tf.square(Y_r - value_output), axis=0)
    
    selected_action_prob = tf.reduce_sum(policy_output * action_index, axis=1) 
    cost_p_1 = tf.math.log(tf.maximum(selected_action_prob, Config.LOG_EPSILON)) * (Y_r - tf.stop_gradient(value_output))
    cost_p_2 = -1 * beta * tf.reduce_sum(tf.math.log(tf.maximum(policy_output, Config.LOG_EPSILON)) * policy_output, axis=1)
    cost_p_1_agg = tf.reduce_sum(cost_p_1, axis=0)
    cost_p_2_agg = tf.reduce_sum(cost_p_2, axis=0)
    cost_p = -(cost_p_1_agg + cost_p_2_agg)

    total_loss = cost_p + value_loss

    return total_loss

class NetworkVPKeras:
    def __init__(self, device, model_name, num_actions):

        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.observation_size = Config.OBSERVATION_SIZE
        self.rotation_size = Config.OBSERVATION_ROTATION_SIZE

        self.observation_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = tf.Variable(Config.BETA_START,dtype=tf.float32)
        self.log_epsilon = Config.LOG_EPSILON

        if Config.LOAD_CHECKPOINT:
            self.model = self.load()

        else:
            self.model = self._build_model()
    
    def _build_model(self):

        self.inputs = tf.keras.Input(shape=(self.observation_size, self.observation_channels))
        self.Conv1 = tf.keras.layers.Conv1D(9, 16, strides=5, activation='relu')(self.inputs)
        self.Conv2 = tf.keras.layers.Conv1D(5, 32, strides=3, activation='relu')(self.Conv1)
        self.flattenlayer = tf.keras.layers.Flatten()(self.Conv2)
        self.Dense1 = tf.keras.layers.Dense(256, activation='relu')(self.flattenlayer)
        
        self.policy_layer = tf.keras.layers.Dense(self.num_actions,activation=None,name="policy_output")(self.Dense1)
        self.softmax_p = (tf.nn.softmax(self.policy_layer) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
        self.q_value_layer = tf.keras.layers.Dense(1, activation=None, name="Qvalue_output")(self.Dense1)

        self.new_model = tf.keras.Model(inputs=self.inputs, outputs=[self.softmax_p, self.q_value_layer])

        self.new_model.compile(
            optimizer= tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, 
                                                  momentum=Config.RMSPROP_MOMENTUM,
                                                  epsilon=Config.RMSPROP_EPSILON, 
                                                  weight_decay=Config.RMSPROP_DECAY),
            loss= loss_function,        
        )
        print(self.new_model.summary())
        return self.new_model 

    # Y_r is target value and a is action index
    @tf.function
    def train(self, x, y_r, a, trainer_id):
        y_true = [a, y_r]

        tf.print("**********************************************")
        tf.print("model training phase with trainer: ", trainer_id)
        tf.print("Shape of x is: ", x.shape)
        
        #if we use self.model.fit() we should pass in a tuple of lists ([x], [policy_layer, q_value_layer]) https://keras.io/guides/functional_api/
        with tf.GradientTape() as tape:
            #forward pass
            logits = self.model(x, training=True)

            #loss_value for batch
            loss_value = loss_function(y_true, logits, self.beta) #loss_value is a tensor of shape (batch_size)
            tf.print("loss during training was calculated as: \n", loss_value, loss_value.shape)

        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        tf.print("**********************************************")

    @tf.function
    def predict_p_and_v(self, x):
        return self.model(x, training=False)

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
        custom_objects = {"loss_function": loss_function}

        with tf.keras.saving.custom_object_scope(custom_objects):
            loaded_model = tf.keras.models.load_model(filepath=filename)

        return loaded_model