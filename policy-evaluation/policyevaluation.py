import numpy as np

from envs.gridworld import GridworldEnv


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


if __name__ == '__main__':
    env = GridworldEnv()
    uniform_policy = np.ones([env.num_states, env.num_actions]) / env.num_actions
    v = policy_eval(uniform_policy, env)

    print("Policy Probability Distribution:\n", uniform_policy, "\n")
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n",
          np.reshape(np.argmax(uniform_policy, axis=1), env.shape), "\n")

    print("Value Function:\n", v, "\n")
    print("Reshaped Grid Value Function:\n", v.reshape(env.shape), "\n")

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array(
        [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
