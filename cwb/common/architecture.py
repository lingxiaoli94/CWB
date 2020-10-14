import tensorflow as tf
from tensorflow import keras
import math
import numpy as np

from .distributions import gaussian_tf

class RFFLayer(keras.layers.Layer):
    def __init__(self, conf):
        super(RFFLayer, self).__init__()

        in_dim = conf['point_dim']
        feat_dim = conf['rff_feat_dim']
        bandwidth = conf['rff_bandwidth']
        seed = conf['rff_seed']
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        w_init = tf.zeros_initializer()
        self.w = tf.Variable(
                initial_value=w_init(shape=(feat_dim), dtype='float32'),
                trainable=True)

        # consistent randomness
        rd_state = np.random.RandomState(seed)
        omegas = rd_state.multivariate_normal(
                np.zeros((in_dim)),
                bandwidth * np.identity(in_dim),
                size=(feat_dim))
        bs = rd_state.uniform(0, 2 * np.pi, size=(feat_dim))
        self.omegas = tf.cast(tf.convert_to_tensor(omegas), tf.float32) # feat_dim x in_dim
        self.bs = tf.cast(tf.convert_to_tensor(bs), tf.float32) # feat_dim
        self.coeff = math.sqrt(2 / feat_dim)

    def call(self, x):
        # x - batch_size x in_dim
        tmp = tf.reduce_sum(tf.expand_dims(self.omegas, 0) * tf.expand_dims(x, 1), -1)
        tmp += tf.expand_dims(self.bs, 0)
        tmp = tf.math.cos(tmp) # batch_size x feat_dim
        zx = self.coeff * tmp # batch_size x feat_dim
        result = tf.reduce_sum(tf.expand_dims(self.w, 0) * zx, -1)
        return result


class PotentialModel(keras.Model):
    def __init__(self, conf):
        super(PotentialModel, self).__init__()

        self.intermediate_layers = []
        model_type = conf['potential_model_type']
        if model_type == 'rff':
            self.final_layer = RFFLayer(
                    in_dim=conf['point_dim'],
                    feat_dim=conf['rff_feat_dim'],
                    bandwidth=conf['rff_bandwidth'],
                    seed=conf['rff_seed'])
        else:
            for layer in conf['potential_nn_layers']:
                self.intermediate_layers.append(keras.layers.Dense(
                    layer,
                    activation='relu'))
            self.final_layer = keras.layers.Dense(
                    1,
                    activation=None)


    def call(self, inputs):
        x = inputs
        for layer in self.intermediate_layers:
            x = layer(x)
        x = self.final_layer(x)
        if x.shape[-1] == 1:
            x = tf.squeeze(x, axis=-1)
        return x


class TransportMapModel(keras.Model):
    def __init__(self, conf):
        super(TransportMapModel, self).__init__()

        self.intermediate_layers = []
        for layer in conf['transport_map_nn_layers']:
            self.intermediate_layers.append(keras.layers.Dense(
                layer,
                activation='relu'))
        self.final_layer = keras.layers.Dense(conf['point_dim'], activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.intermediate_layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


