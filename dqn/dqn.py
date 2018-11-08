import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from viz import viz
from collections import deque, namedtuple

VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor:
    def __init__(self):
        with tf.variable_scope('state_processor'):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess: tf.Session, state: Tensor) -> Tensor:
        return sess.run(self.output, {self.input_state: state})


class Estimator:
    def __init__(self, scope='estimator', summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name='y')
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        conv1 = tf.layers.conv2d(X, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)
        flattened = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(flattened, 512)
        self.predictions = tf.layers.dense(fc1, len(VALID_ACTIONS))

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.contrib.framework.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.histogram('loss_hist', self.losses),
            tf.summary.histogram('q_values_hist', self.predictions),
            tf.summary.scalar('max_q_value', tf.reduce_max(self.predictions))
        ])

    def predict(self, sess: tf.Session, s: np.ndarray) -> Tensor:
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self,
               sess: tf.Session,
               s: np.ndarray,
               a: np.ndarray,
               y: np.ndarray) -> Tensor:

        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


if __name__ == '__main__':
    env = gym.envs.make('Breakout-v0')

    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    e = Estimator(scope='test')
    sp = StateProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        observation = env.reset()

        observation_p = sp.process(sess, observation)
        observation = np.stack([observation_p] * 4, axis=2)
        observations = np.array([observation] * 2)

        print(e.predict(sess, observations))

        y = np.array([10.0, 10.0])
        a = np.array([1, 3])
        print(e.update(sess, observations, a, y))
