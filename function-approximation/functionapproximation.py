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
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    stats = EpisodeStats(length=np.zeros(num_episodes), reward=np.zeros(num_episodes))

    for i in tqdm(range(num_episodes)):

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.reward[i - 1]

        # Reset the environment and pick the first action
        state = env.reset()

        # Only used for SARSA, not Q-Learning
        next_action = None

        # One step in the environment
        for t in itertools.count():

            # Choose an action to take
            # If we're using SARSA we already decided in the previous step
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.reward[i] += reward
            stats.length[i] = t

            # TD Update
            q_values_next = estimator.predict(next_state)

            # Use this code for Q-Learning
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)

            # Use this code for SARSA TD Target for on policy-training:
            # next_action_probs = policy(next_state)
            # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            # td_target = reward + discount_factor * q_values_next[next_action]

            # Update the function approximator using our target
            estimator.update(state, action, td_target)

            if done:
                break

            state = next_state

    return stats


if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')

    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    num_components = 100
    featurizer = FeatureUnion([
        ('rbf1', RBFSampler(gamma=5.0, n_components=num_components)),
        ('rbf2', RBFSampler(gamma=2.0, n_components=num_components)),
        ('rbf3', RBFSampler(gamma=1.0, n_components=num_components)),
        ('rbf4', RBFSampler(gamma=0.5, n_components=num_components))
    ])
    featurizer.fit(scaler.transform(observation_examples))

    estimator = Estimator()

    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    stats = q_learning(env, estimator, 1000, epsilon=0.0)

    viz.plot_cost_to_go_mountain_car(env, estimator)
    viz.plot_episode_stats(stats, smoothing_window=25)
