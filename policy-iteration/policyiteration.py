from typing import Tuple, Callable

import numpy as np

from envs.gridworld import GridworldEnv

PolicyEvalArgs = Callable[[np.array, GridworldEnv], np.array]


def policy_eval(policy: np.array,
                env: GridworldEnv,
                discount_factor: float = 1.0,
                theta: float = 1e-9) -> np.array:
    V = np.zeros(env.num_states)
    while True:
        delta = 0
        for s in range(env.num_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.p[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def one_step_lookahead(state: int,
                       V: np.ndarray,
                       discount_factor: float = 1.0) -> np.array:
    A = np.zeros(env.num_actions)
    for a in range(env.num_actions):
        for prob, next_state, reward, done in env.p[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


def policy_improvement(env: GridworldEnv,
                       policy_eval_fn: PolicyEvalArgs) -> Tuple[np.array, np.array]:
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    while True:
        V = policy_eval_fn(policy, env)
        policy_stable = True

        for s in range(env.num_states):
            chosen_action = np.argmax(policy[s])

            action_values = one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            if chosen_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.num_actions)[best_action]

        if policy_stable:
            return policy, V


if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = policy_improvement(env, policy_eval)

    print("Policy Probability Distribution:\n", policy, "\n")
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n",
          np.reshape(np.argmax(policy, axis=1), env.shape), "\n")

    print("Value Function:\n", v, "\n")
    print("Reshaped Grid Value Function:\n", v.reshape(env.shape), "\n")

    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
