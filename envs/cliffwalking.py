from typing import List, Tuple

import numpy as np
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord: np.array) -> np.array:
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self,
                                   current: Tuple,
                                   delta: List) -> Tuple[float, np.array, float, bool]:
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        reward = -100.0 if self._cliff[tuple(new_position)] else -1.0
        is_done = self._cliff[tuple(new_position)] or (tuple(new_position) == (3, 11))
        return 1.0, new_state, reward, is_done

    def __init__(self):
        self.shape = (4, 12)
        num_states = int(np.prod(self.shape))
        num_actions = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        p = {}
        for s in range(num_states):
            position = np.unravel_index(s, self.shape)
            p[s] = {}
            p[s][UP] = self._calculate_transition_prob(position, [1, 0])
            p[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            p[s][DOWN] = self._calculate_transition_prob(position, [-1, 0])
            p[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # We always start in state (3, 0)
        isd = np.zeros(num_states)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0
        super(CliffWalkingEnv, self).__init__(num_states, num_actions, p, isd)
