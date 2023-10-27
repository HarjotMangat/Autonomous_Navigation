import numpy as np
import tensorflow as tf
import os
import multiprocessing

from Config import Config

#@tf.keras.utils.register_keras_serializable(package='NetworkVPKeras', name="loss_function1")
def loss_function(y_true, y_pred):
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

    #tf.print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    action_index, Y_r = y_true #action_index is a tensor of shape (batch_size, num_actions), Y_r is a tensor of shape (batch_size, 1)
    #tf.print("action_index is: ", action_index)
    #tf.print("Y_r is: ", Y_r)
    policy_output, value_output = y_pred # policy_output is a tensor of shape (batch_size, num_actions), value_output is a tensor of shape (batch_size, 1)
    #tf.print("policy_output is: ", policy_output)
    #tf.print("value_output is: ", value_output)
    #tf.print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    value_loss = 0.5 * tf.math.squared_difference(Y_r, value_output)

    #policy_loss = -tf.reduce_sum(tf.math.log(policy_output + 1e-10) * action_index, axis=1)
    #total_loss = policy_loss + (0.5 * value_loss) - (0.01 * policy_output)

    selected_action_prob = tf.reduce_sum(policy_output * action_index, axis=1) 
    cost_p_1 = tf.math.log(tf.maximum(selected_action_prob, Config.LOG_EPSILON)) * (Y_r - tf.stop_gradient(value_output))
    cost_p_2 = -1 * Config.BETA_START * tf.reduce_sum(tf.math.log(tf.maximum(policy_output, Config.LOG_EPSILON)) * policy_output, axis=1)

    cost_p_1_agg = tf.reduce_sum(cost_p_1, axis=0)
    cost_p_2_agg = tf.reduce_sum(cost_p_2, axis=0)
    cost_p = -(cost_p_1_agg + cost_p_2_agg)

    total_loss = cost_p + value_loss

    return total_loss

class NetworkVPKeras:
    def __init__(self, device, model_name, num_actions):
        #physical_devices = tf.config.list_physical_devices('GPU')
        #try:
        #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
        #    print("worked")
        #except:
        #    print("didn't work")
            # Invalid device or cannot modify virtual devices once initialized.
        #    pass
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions
        self.lock = multiprocessing.Lock()

        self.observation_size = Config.OBSERVATION_SIZE
        self.rotation_size = Config.OBSERVATION_ROTATION_SIZE

        self.observation_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        if Config.LOAD_CHECKPOINT:
            self.model = self.load()

        else:
            self.model = self._build_model()
        

    
    def _build_model(self):

        #self.inputs = tf.keras.Input(shape=(None, self.observation_size, self.observation_channels))
        self.inputs = tf.keras.Input(shape=(self.observation_size, self.observation_channels))
        self.Conv1 = tf.keras.layers.Conv1D(9, 16, strides=4, activation='relu')(self.inputs)
        self.Conv2 = tf.keras.layers.Conv1D(5, 32, strides=3, activation='relu')(self.Conv1)
        self.flattenlayer = tf.keras.layers.Flatten()(self.Conv2)
        self.Dense1 = tf.keras.layers.Dense(256, activation='relu')(self.flattenlayer)
        
        self.policy_layer = tf.keras.layers.Dense(self.num_actions, activation=tf.keras.activations.softmax, name="policy_output")(self.Dense1)
        self.Q_value_layer = tf.keras.layers.Dense(1, activation=None, name="Qvalue_output")(self.Dense1)

        self.new_model = tf.keras.Model(inputs=self.inputs, outputs=[self.policy_layer, self.Q_value_layer])

        self.new_model.compile(
            optimizer= tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, 
                                                  momentum=Config.RMSPROP_MOMENTUM,
                                                  epsilon=Config.RMSPROP_EPSILON, 
                                                  weight_decay=Config.RMSPROP_DECAY),
            loss= loss_function,
            #loss={'policy_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            #      'Qvalue_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
        
        )
        print(self.new_model.summary())
        return self.new_model 

    # Y_r is target value and a is action index
    # Maybe set this up as tf.function and pass x, y_r, a as tensors
    @tf.function
    def train(self, x, y_r, a, trainer_id):

        #self.lock.acquire()
        y_true = [a, y_r]

        tf.print("**********************************************")
        tf.print("model training phase with trainer: ", trainer_id)
        #print("model training phase with trainer: ", trainer_id)
        #print(x)
        tf.print("Shape of x is: ", x.shape)
        #print(y_r)
        #print("Shape of y_r is: ", y_r.shape)
        #print(a)
        #print("Shape of a is: ", a.shape)
        
        #if we use self.model.fit() we should pass in a tuple of lists ([x], [policy_layer, Q_value_layer]) https://keras.io/guides/functional_api/
        with tf.GradientTape() as tape:

            #forward pass
            logits = self.model(x, training=True)

            #loss_value for batch
            #tf.print("y_true is: \n", y_true) #y_true is [actions (batch_size, num_actions), rewards (batch_size, 1)]
            #tf.print("logits are: \n", logits) #logits is [policy_layer (batch_size, num_actions), Q_value_layer (batch_size, 1)]
            loss_value = loss_function(y_true, logits) #loss_value is a tensor of shape (batch_size, batch_size)
            tf.print("loss during training was calculated as: \n", loss_value, loss_value.shape)

        gradients = tape.gradient(loss_value, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        tf.print("**********************************************")
        #self.lock.release()

    @tf.function
    def predict_p_and_v(self, x):

        #self.lock.acquire()
        #tf.print("got to the model prediction phase")
        #print("value of x is: ", x)
        #print("shape of x is: ", x.shape)
        #result = self.model(x, training=False)
        #self.lock.release()
        
        #return result
        return self.model(x, training=False)

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d.keras' % (self.model_name, episode)
    
    def save(self, episode):
        #self.saver.save(self.sess, self._checkpoint_filename(episode))
        tf.keras.models.save_model(model=self.model, filepath=self._checkpoint_filename(episode=episode))

    def load(self):
        print("Loading model now")
        filename = (os.path.dirname(self._checkpoint_filename(episode=0)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        print("filename is: ", filename)
        custom_objects = {"loss_function": loss_function}

        with tf.keras.saving.custom_object_scope(custom_objects):
            loaded_model = tf.keras.models.load_model(filepath=filename)

        #loaded_model = tf.keras.models.load_model(filename)
        #self.saver.restore(self.sess, filename)
        #return self._get_episode_from_filename(filename)
        return loaded_model