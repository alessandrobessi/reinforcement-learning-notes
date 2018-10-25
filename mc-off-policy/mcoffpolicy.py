from collections import defaultdict, namedtuple
from typing import Dict, Tuple, Callable

import numpy as np
from tqdm import tqdm

from envs.blackjack import BlackjackEnv
from viz import viz

Step = namedtuple('Step', ['state', 'action', 'reward'])


def create_random_policy(num_actions: int) -> Callable:
    A = np.ones(num_actions, dtype=float) / num_actions

    def policy_fn() -> np.array:
        return A

    return policy_fn


def create_greedy_policy(Q: Dict) -> Callable:
    def policy_fn(state: int) -> np.array:
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A

    return policy_fn


def mc_control_importance_sampling(env: BlackjackEnv,
                                   num_episodes: float,
                                   behavior_policy: Callable,
                                   discount_factor: float = 1.0) -> Tuple[Dict, Callable]:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    target_policy = create_greedy_policy(Q)

    for _ in tqdm(range(int(num_episodes))):
        episode = []
        state = env.reset()
        is_over = False
        while not is_over:
            probs = behavior_policy()
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, is_over, _ = env.step(action)
            episode.append(Step(state, action, reward))
            state = next_state

        g = 0.0
        w = 1.0
        for ep in episode[::-1]:
            g = discount_factor * g + ep.reward
            C[ep.state][ep.action] += w
            Q[ep.state][ep.action] += (w / C[ep.state][ep.action]) * (g - Q[ep.state][ep.action])
            if ep.action != np.argmax(target_policy(ep.state)):
                break
            w = w * 1.0 / behavior_policy()[ep.action]

    return Q, target_policy


if __name__ == '__main__':
    env = BlackjackEnv()

    random_policy = create_random_policy(env.action_space.n)
    Q, policy = mc_control_importance_sampling(env,
                                               num_episodes=1e6,
                                               behavior_policy=random_policy)

    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    viz.plot_value_function(V, title="Optimal Value Function")
