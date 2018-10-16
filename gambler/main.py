import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

with open(os.path.join(os.getcwd(), 'gambler.md'), encoding='utf-8') as f:
    description = f.read()

CAPITAL = 100
ACTION_SPACE = CAPITAL + 1


def one_step_lookahead(p: float,
                       gamma: float,
                       s: int,
                       V: np.array,
                       rewards: np.array) -> np.array:
    A = np.zeros(ACTION_SPACE)
    stakes = range(1, min(s, CAPITAL - s) + 1)
    for a in stakes:
        A[a] = p * (rewards[s + a] + V[s + a] * gamma) + (1 - p) * (rewards[s - a] + V[s - a] *
                                                                    gamma)
    return A


def value_iteration(p: float,
                    theta: float = 1e-32,
                    gamma: float = 1.0) -> Tuple[np.array, np.array]:
    rewards = np.zeros(ACTION_SPACE)
    rewards[CAPITAL] = 1

    V = np.zeros(ACTION_SPACE)
    policy = np.zeros(CAPITAL)

    while True:
        delta = 0
        for s in range(1, CAPITAL):
            A = one_step_lookahead(p, gamma, s, V, rewards)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value

        if delta < theta:
            break

    policy = np.zeros(CAPITAL)
    for s in range(1, CAPITAL):
        A = one_step_lookahead(p, gamma, s, V, rewards)
        policy[s] = np.argmax(A)

    return policy, V


if __name__ == '__main__':
    print(description)

    policy, V = value_iteration(0.4)

    print("Optimized Policy")
    print(policy)
    print()

    print("Optimized Value Function")
    print(V)

    x = range(100)
    y = V[:100]

    plt.plot(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.show()

    x = range(100)
    y = policy

    plt.bar(x, y, align='center', alpha=0.5)
    plt.xlabel('Capital')
    plt.ylabel('Final Policy (stake)')
    plt.show()
