import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

with open(os.path.join(os.getcwd(), 'gamblerproblem.md'), encoding='utf-8') as f:
    description = f.read()

CAPITAL = 100
ACTION_SPACE = CAPITAL + 1


def show_optimal_policy(p: np.array) -> None:
    plt.bar(range(CAPITAL), p, align='center', alpha=0.5)
    plt.xlabel('Capital')
    plt.ylabel('Final Policy (stake)')
    plt.show()


def show_value_estimates(v) -> None:
    plt.plot(range(CAPITAL), v[:CAPITAL])
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.show()


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
                    theta: float = 1e-300,
                    gamma: float = 1.0) -> Tuple[np.array, np.array]:
    rewards = np.zeros(ACTION_SPACE)
    rewards[CAPITAL] = 1

    V = np.zeros(ACTION_SPACE)

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

    print(policy)

    return policy, V


if __name__ == '__main__':
    print(description)

    policy, V = value_iteration(0.4)

    print("Optimal Policy")
    print(policy)
    print()

    print("OPtimal Value Function")
    print(V)

    show_optimal_policy(policy)
    show_value_estimates(V)
