from collections import defaultdict, namedtuple
from typing import Dict, Tuple, Callable

import numpy as np
from tqdm import tqdm

from envs.blackjack import BlackjackEnv
from viz import viz

Step = namedtuple('Step', ['state', 'action', 'reward'])


def make_epsilon_greedy_policy(Q: Dict,
                               epsilon: float,
                               num_actions: int) -> Callable[[int], np.array]:
    def policy_fn(state: int) -> np.array:
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def mc_control_epsilon_greedy(env: BlackjackEnv,
                              num_episodes: float,
                              discount_factor: float = 1.0,
                              epsilon: float = 0.1) -> Tuple[Dict, Callable[[int], np.array]]:
    rewards_sum = defaultdict(float)
    rewards_count = defaultdict(float)

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for _ in tqdm(range(int(num_episodes))):

        episode = []
        state = env.reset()
        is_over = False
        while not is_over:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, is_over, _ = env.step(action)
            episode.append(Step(state, action, reward))
            state = next_state

        sa_in_episode = set([(tuple(ep.state), ep.action) for ep in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            first_visit = next(i for i, ep in enumerate(episode)
                               if ep.state == state and ep.action == action)
            g = sum([ep.reward * (discount_factor ** i) for i, ep in
                     enumerate(episode[first_visit:])])
            rewards_sum[sa_pair] += g
            rewards_count[sa_pair] += 1.0
            Q[state][action] = rewards_sum[sa_pair] / rewards_count[sa_pair]

    return Q, policy


if __name__ == '__main__':
    env = BlackjackEnv()
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=1e6, epsilon=0.1)

    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    viz.plot_value_function(V, title='Optimal Value Function')
