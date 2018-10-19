from collections import defaultdict
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from envs.blackjack import BlackjackEnv


def plot_surface(x: np.array, y: np.array, z: np.array, title: str) -> None:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                              cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surface)
    plt.show()


def plot_value_function(v: Dict[Tuple, float],
                        title: str = "Value Function") -> None:
    min_x = min(k[0] for k in v.keys())
    max_x = max(k[0] for k in v.keys())
    min_y = min(k[1] for k in v.keys())
    max_y = max(k[1] for k in v.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)

    x, y = np.meshgrid(x_range, y_range)

    z_no_ace = np.apply_along_axis(lambda _: v[(_[0], _[1], False)], 2, np.dstack([x, y]))
    z_ace = np.apply_along_axis(lambda _: v[(_[0], _[1], True)], 2, np.dstack([x, y]))

    plot_surface(x, y, z_no_ace, "{} (No Usable Ace)".format(title))
    plot_surface(x, y, z_ace, "{} (Usable Ace)".format(title))


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
            first_occurrence_idx = next(i for i, ep in enumerate(episode) if ep[0] == state)
            g = sum([ep[2] * (discount_factor ** i) for i, ep in
                     enumerate(episode[first_occurrence_idx:])])
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
    plot_value_function(V_1M, title="Value Function after 1M Steps")
