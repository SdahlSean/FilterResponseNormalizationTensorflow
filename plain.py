import tensorflow as tf
import tensorflow.keras as k


class FRN(k.layers.Layer):
    def __init__(self, eps=1e-6):
        super(FRN, self).__init__()
        self.eps = eps

    def build(self, input_shape):
        self.beta = self.add_variable('beta', shape=(1, 1, 1, input_shape[3]))
        self.gamma = self.add_variable('gamma', shape=(1, 1, 1, input_shape[3]))
        self.tau = self.add_variable('tau', shape=(1, 1, 1, input_shape[3]))

    def call(self, x):
        v2 = tf.math.reduce_mean(x, (1, 2), True)
        xhat = x/tf.math.sqrt(v2 + self.eps)
        y = self.gamma * xhat + self.beta
        z = tf.math.maximum(y, self.tau)
        return z
