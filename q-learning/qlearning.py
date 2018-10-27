import itertools
import numpy as np
from collections import defaultdict, namedtuple
from typing import Dict, Callable, Tuple
from tqdm import tqdm
from envs.windygridworld import WindyGridworldEnv
from viz import viz

EpisodeStats = namedtuple('Stats', ['length', 'reward'])


def make_epsilon_greedy_policy(Q: Dict,
                               epsilon: float,
                               num_actions: int) -> Callable[[int], np.array]:
    def policy_fn(state: int):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env: WindyGridworldEnv,
               num_episodes: float,
               discount: float = 1.0,
               alpha: float = 0.5,
               epsilon: float = 0.1) -> Tuple[Dict, EpisodeStats]:
    num_episodes = int(num_episodes)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    stats = EpisodeStats(np.zeros(num_episodes), np.zeros(num_episodes))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in tqdm(range(num_episodes)):
        state = env.reset()
        for t in itertools.count():

            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            stats.reward[i] += reward
            stats.length[i] = t

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


if __name__ == '__main__':
    env = WindyGridworldEnv()

    Q, stats = q_learning(env, 1e4)
    viz.plot_episode_stats(stats)
