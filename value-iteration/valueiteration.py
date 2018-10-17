import numpy as np

from envs.gridworld import GridworldEnv


def deterministic_policy(env: GridworldEnv,
                         V: np.array) -> np.array:
    policy = np.zeros([env.num_states, env.num_actions])
    for s in range(env.num_states):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy


def one_step_lookahead(state: int,
                       V: np.ndarray,
                       discount_factor: float = 1.0) -> np.array:
    A = np.zeros(env.num_actions)
    for a in range(env.num_actions):
        for prob, next_state, reward, done in env.p[state][a]:
            A[a] += prob * (reward + discount_factor * V[next_state])
    return A


def value_iteration(env: GridworldEnv,
                    theta: float = 1e-12) -> np.array:
    V = np.zeros(env.num_states)
    while True:
        delta = 0
        for s in range(env.num_states):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    return V


if __name__ == '__main__':
    env = GridworldEnv()

    v = value_iteration(env)
    policy = deterministic_policy(env, v)

    print("Policy Probability Distribution:\n", policy, "\n")
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n",
          np.reshape(np.argmax(policy, axis=1), env.shape), "\n")
    print("Value Function:\n", v, "\n")
    print("Reshaped Grid Value Function:\n", v.reshape(env.shape), "\n")

    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
