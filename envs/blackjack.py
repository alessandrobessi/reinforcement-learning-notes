from typing import List, Tuple, Dict

import gym
from gym import spaces
from gym.utils import seeding


class BlackjackEnv(gym.Env):
    deck = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10)

    @staticmethod
    def compute_reward(player_score: int, dealer_score: int) -> int:
        return 1 if player_score > dealer_score else -1 if player_score < dealer_score else 0

    @staticmethod
    def draw_card(np_random: seeding.np_random) -> int:
        return np_random.choice(BlackjackEnv.deck)

    @staticmethod
    def draw_hand(np_random: seeding.np_random) -> List:
        return [BlackjackEnv.draw_card(np_random), BlackjackEnv.draw_card(np_random)]

    @staticmethod
    def usable_ace(hand: List) -> bool:
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def sum_hand(hand: List) -> int:
        if BlackjackEnv.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def is_bust(hand: List) -> bool:
        return BlackjackEnv.sum_hand(hand) > 21

    @staticmethod
    def score(hand: List) -> int:
        return 0 if BlackjackEnv.is_bust(hand) else BlackjackEnv.sum_hand(hand)

    @staticmethod
    def is_natural(hand: List) -> bool:
        return tuple(sorted(hand)) == (1, 10)

    def __init__(self, natural: bool = False):
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self._seed()
        self.natural = natural
        self._reset()
        self.num_actions = 2

    def reset(self):
        return self._reset()

    def step(self, action: int) -> Tuple[Tuple, int, bool, Dict]:
        return self._step(action)

    def _seed(self, seed: int = None) -> List:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action: int) -> Tuple[Tuple, int, bool, Dict]:
        assert self.action_space.contains(action)
        if action:
            self.player.append(BlackjackEnv.draw_card(self.np_random))
            if BlackjackEnv.is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:
            done = True
            while BlackjackEnv.sum_hand(self.dealer) < 17:
                self.dealer.append(BlackjackEnv.draw_card(self.np_random))
            reward = BlackjackEnv.compute_reward(BlackjackEnv.score(self.player),
                                                 BlackjackEnv.score(self.dealer))
            if self.natural and BlackjackEnv.is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> Tuple[int, int, bool]:
        return (
            BlackjackEnv.sum_hand(self.player), self.dealer[0],
            BlackjackEnv.usable_ace(self.player))

    def _reset(self) -> Tuple[int, int, bool]:
        self.dealer = BlackjackEnv.draw_hand(self.np_random)
        self.player = BlackjackEnv.draw_hand(self.np_random)

        while BlackjackEnv.sum_hand(self.player) < 12:
            self.player.append(BlackjackEnv.draw_card(self.np_random))

        return self._get_obs()
