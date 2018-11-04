import gym
import itertools
import numpy as np
from typing import Union, Callable
from collections import namedtuple
from viz import viz
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

EpisodeStats = namedtuple('Stats', ['length', 'reward'])


class Estimator:

    @staticmethod
    def featurize_state(state: np.ndarray) -> np.ndarray:
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def __init__(self):
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant', tol=1e-3)
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def predict(self, s: np.ndarray, a: int = None) -> Union[int, np.array]:
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        return self.models[a].predict([features])[0]

    def update(self, s: np.ndarray, a: int, y: int) -> None:
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator: Estimator,
                               epsilon: float,
                               num_actions: int) -> Callable[[np.ndarray], np.array]:
    def policy_fn(state: np.ndarray) -> np.array:
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(state)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env,
               estimator: Estimator,
               num_episodes: int,
               discount_factor: float = 1.0,
               epsilon: float = 0.1,
               epsilon_decay: float = 1.0) -> EpisodeStats:
    stats = EpisodeStats(length=np.zeros(num_episodes), reward=np.zeros(num_episodes))

    for i in tqdm(range(num_episodes)):

        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i, env.action_space.n)

        state = env.reset()

        next_action = None

        for t in itertools.count():

            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            next_state, reward, done, _ = env.step(action)

            stats.reward[i] += reward
            stats.length[i] = t

            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)
            estimator.update(state, action, td_target)

            if done:
                break

            state = next_state

    return stats


if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')

    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    num_components = 100
    featurizer = FeatureUnion([
        ('rbf1', RBFSampler(gamma=5.0, n_components=num_components)),
        ('rbf2', RBFSampler(gamma=2.0, n_components=num_components)),
        ('rbf3', RBFSampler(gamma=1.0, n_components=num_components)),
        ('rbf4', RBFSampler(gamma=0.5, n_components=num_components))
    ])
    featurizer.fit(scaler.transform(observation_examples))

    estimator = Estimator()
    stats = q_learning(env, estimator, 1000, epsilon=0.0)

    viz.plot_cost_to_go_mountain_car(env, estimator)
    viz.plot_episode_stats(stats, smoothing_window=25)
