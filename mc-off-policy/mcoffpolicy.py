from collections import defaultdict
from typing import Dict, List, Tuple, Union, Callable

import numpy as np
from tqdm import tqdm

from envs.blackjack import BlackjackEnv
from viz import viz


def create_random_policy(num_actions: int) -> Callable:
    A = np.ones(num_actions, dtype=float) / num_actions

    def policy_fn(state: int) -> np.array:
        return A

    return policy_fn


def create_greedy_policy(Q: Dict) -> Callable:
    def policy_fn(state: int):
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
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)

    for _ in tqdm(range(int(num_episodes))):
        episode = []
        state = env.reset()
        is_over = False
        while not is_over:
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, is_over, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        g = 0.0
        # The importance sampling ratio (the weights of the returns)
        w = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            g = discount_factor * g + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += w
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (w / C[state][action]) * (g - Q[state][action])
            # If the action taken by the behavior policy is not the action
            # taken by the target policy the probability will be 0 and we can break
            if action != np.argmax(target_policy(state)):
                break
            w = w * 1.0 / behavior_policy(state)[action]

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
