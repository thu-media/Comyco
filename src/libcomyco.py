import numpy as np
import tensorflow as tf
import tflearn
import time
import warnings
import pool

RAND_RANGE = 1000
FEATURE_NUM = 128

# you can use mish active function instead
# the total performance will be improved a little bit.
def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))


class libcomyco(object):
    def __init__(self, sess, S_INFO, S_LEN, A_DIM, LR_RATE=1e-4, ID=1):
        self.pool_ = pool.pool()
        self.sess = sess
        self.S_INFO = S_INFO
        self.S_LEN = S_LEN
        self.A_DIM = A_DIM
        self.s_name = 'actor/' + str(ID)

        self.inputs, self.out = self.create_network()

        self.real_out = tf.clip_by_value(self.out, 1e-4, 1. - 1e-4)

        self.y_ = tf.placeholder(shape=[None, A_DIM], dtype=tf.float32)

        # you can use any loss you want
        # self.core_net_loss = tflearn.objectives.mean_square(
        #     self.real_out, self.y_) + 1e-3 * tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)))
        # self.core_net_loss = -tf.reduce_sum(self.y_ * tf.log(self.real_out)) + 1e-3 * tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)))
        # Note: here is a minor mistake in the camera ready paper, the fomular of eq(4) should be l_comyco = -log(pi) * A* - \beta * entropy
        self.core_net_loss = tflearn.objectives.categorical_crossentropy(
            self.real_out, self.y_) + 1e-3 * tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)))

        self.core_train_op = tf.train.AdamOptimizer(
            learning_rate=LR_RATE).minimize(self.core_net_loss)

        self.saver = tf.train.Saver()  # save neural net parameters

    def create_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(
                shape=[None, self.S_INFO, self.S_LEN])
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.A_DIM], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.conv_1d(
                inputs[:, 5:6, :self.A_DIM], FEATURE_NUM, 4, activation='relu')
            split_6 = tflearn.fully_connected(
                inputs[:, 6:7, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)
            split_5_flat = tflearn.flatten(split_5)

            merge_net = tf.stack(
                [split_0, split_1, split_2_flat,
                 split_3_flat, split_4_flat, split_5_flat, split_6], axis=-1)
            # shuffle to fit gru layer
            merge_net = tf.transpose(merge_net, [0, 2, 1])
            dense_net_0 = tflearn.gru(
                merge_net, FEATURE_NUM, activation='relu')

            out = tflearn.fully_connected(
                dense_net_0, self.A_DIM, activation='softmax')

            return inputs, out

    def predict(self, state):
        action_prob = self.sess.run(self.real_out, feed_dict={
            self.inputs: np.reshape(state, (-1, self.S_INFO, self.S_LEN))
        })
        # randomly picks an action
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(
            1, RAND_RANGE) / float(RAND_RANGE)).argmax()

        return action_prob, bit_rate

    def loss(self, state, action_real_vec):
        loss_ = self.sess.run(self.core_net_loss, feed_dict={
            self.inputs: np.reshape(state, (-1, self.S_INFO, self.S_LEN)),
            self.y_: np.reshape(action_real_vec, (-1, self.A_DIM))
        })
        return loss_

    def submit(self, state, action_real_vec):
        self.pool_.submit(state, action_real_vec)

    def train(self):
        training_s_batch, training_a_batch = self.pool_.get()
        if training_s_batch.shape[0] > 0:
            self.sess.run(self.core_train_op, feed_dict={
                self.inputs: np.array(training_s_batch),
                self.y_: np.array(training_a_batch)
            })

    def save(self, filename):
        self.saver.save(self.sess, filename)

    def load(self, filename):
        self.saver.restore(self.sess, filename)

    def compute_entropy(self, x):
        """
        Given vector x, computes the entropy
        H(x) = - sum( p * log(p))
        """
        H = 0.0
        x = np.clip(x, 1e-5, 1.)
        for i in range(len(x)):
            if 0 < x[i] < 1:
                H -= x[i] * np.log(x[i])
        return H
