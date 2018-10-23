from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from envs.blackjack import BlackjackEnv
from viz import viz


def mc_prediction(policy: np.array,
                  env: BlackjackEnv,
                  num_episodes: Union[int, float],
                  discount_factor: float = 1.0) -> Dict[Tuple, float]:
    rewards_sum = defaultdict(float)
    rewards_count = defaultdict(float)
    v = defaultdict(float)

    for _ in tqdm(range(int(num_episodes))):

        episode = []
        state = env.reset()
        is_over = False
        while not is_over:
            action = policy(state)
            next_state, reward, is_over, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        states_in_episode = set([tuple(ep[0]) for ep in episode])
        for state in states_in_episode:
            first_visit_idx = next(i for i, ep in enumerate(episode) if ep[0] == state)
            g = sum([ep[2] * (discount_factor ** i) for i, ep in
                     enumerate(episode[first_visit_idx:])])
            rewards_sum[state] += g
            rewards_count[state] += 1.0
            v[state] = rewards_sum[state] / rewards_count[state]

    return v


def dumb_policy(state: List) -> int:
    player_score, dealer_score, usable_ace = state
    return 0 if player_score >= 18 else 1


if __name__ == '__main__':
    env = BlackjackEnv()

    V_1M = mc_prediction(dumb_policy, env, num_episodes=1e6)
    viz.plot_value_function(V_1M, title='Value Function after 1M Steps')
