import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

from viz import viz
from collections import deque, namedtuple

if __name__ == '__main__':
    env = gym.envs.make('Breakout-v0')
