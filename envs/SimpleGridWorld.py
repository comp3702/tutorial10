# Directions
import random
from typing import Dict, Tuple, Optional

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {UP: 'U', DOWN: 'D', LEFT: 'L', RIGHT: 'R'}

EXIT_STATE = (-1, -1)

EPSILON = 0.0001

class SimpleGridWorld:
    def __init__(self, cols: int = 4, rows: int = 3, p: float = 0.8,
                 gamma: float = 0.9,
                 rewards: Optional[Dict[Tuple[int, int], int]] = None,
                 obstacles: Tuple[Tuple[int, int]] = ((1, 1),)):
        self.last_col = cols - 1
        self.last_row = rows - 1

        self.p = p
        self.alt_p = (1 - p) / 2

        self.actions = [UP, DOWN, LEFT, RIGHT]

        if rewards is None:
            self.rewards = {(3, 1): -100, (3, 2): 1}
        else:
            self.rewards = rewards

        self.gamma = gamma

        states = list( (col, row) for row in range(rows) for col in range(cols) )
        # states.append(EXIT_STATE)
        for obstacle in obstacles:
            states.remove(obstacle)
        self.states = tuple(states)

        self.obstacles = obstacles

    def attempt_move(self, s: Tuple[int, int], a: int) -> Tuple[int, int]:
        """ Attempts to move the agent from state s via action a.

            Parameters:
                s: The current state.
                a: The *actual* action performed (as opposed to the chosen
                   action; i.e. you do not need to account for non-determinism
                   in this method).
            Returns: the state resulting from performing action a in state s.
        """
        col, row = s

        # Check borders
        if a == RIGHT and col < self.last_col:
            col += 1
        elif a == LEFT and col > 0:
            col -= 1
        # indexed at bottom left!!!!! not top
        elif a == UP and row < self.last_row:
            row += 1
        elif a == DOWN and row > 0:
            row -= 1

        result = (col, row)

        # Check obstacle cells
        if result in self.obstacles:
            result = s

        return result

    def stoch_action(self, action: int) -> Dict[int, float]:
        """ Returns the probabilities with which each action will actually occur,
            given that action a was requested.

        Parameters:
            action: The action requested by the agent.

        Returns:
            The probability distribution over actual actions that may occur.
        """
        if action == RIGHT:
            return {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        elif action == UP:
            return {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        elif action == LEFT:
            return {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        return {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], int, bool]:
        """
        :param state: current state
        :param action: action to be taken
        :return: tuple of (next_state, reward, episode_done)
        """
        stochastic_actions = self.stoch_action(action)
        random_action = random.choices(population=list(stochastic_actions.keys()),
                                       weights=list(stochastic_actions.values()))[0]

        next_state = self.attempt_move(state, random_action)

        if next_state in self.rewards.keys():
            return next_state, self.rewards[next_state], True
        else:
            return next_state, 0, False

    def reset(self) -> Tuple[int, int]:
        """
        randomly initializes the environment state - i.e. picks a random state to start from
        :return: tuple of (next_state, reward, episode_done)
        """
        return random.choice(self.states)