from typing import Tuple, List, Union

import numpy as np
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape: Union[Tuple, List] = (4, 4)):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        self.num_states = int(np.prod(shape))
        self.num_actions = 4

        max_y = shape[0]
        max_x = shape[1]

        p = {}
        grid = np.arange(self.num_states).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            p[s] = {a: [] for a in range(self.num_actions)}

            reward = 0.0 if self.is_done(s) else -1.0

            # We're stuck in a terminal state
            if self.is_done(s):
                p[s][UP] = [(1.0, s, reward, True)]
                p[s][RIGHT] = [(1.0, s, reward, True)]
                p[s][DOWN] = [(1.0, s, reward, True)]
                p[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - max_x
                ns_right = s if x == (max_x - 1) else s + 1
                ns_down = s if y == (max_y - 1) else s + max_x
                ns_left = s if x == 0 else s - 1
                p[s][UP] = [(1.0, ns_up, reward, self.is_done(ns_up))]
                p[s][RIGHT] = [(1.0, ns_right, reward, self.is_done(ns_right))]
                p[s][DOWN] = [(1.0, ns_down, reward, self.is_done(ns_down))]
                p[s][LEFT] = [(1.0, ns_left, reward, self.is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(self.num_states) / self.num_states

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.p = p

        super(GridworldEnv, self).__init__(self.num_states, self.num_actions, self.p, isd)

    def is_done(self, s: int) -> bool:
        return True if s == 0 or s == (self.num_states - 1) else False
