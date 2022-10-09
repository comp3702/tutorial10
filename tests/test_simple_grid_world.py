import unittest
from typing import Tuple, Set

from envs.SimpleGridWorld import SimpleGridWorld, RIGHT, LEFT, DOWN, UP


class TestSimpleGridWorld(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SimpleGridWorld()

    def test_init(self):
        self.assertEqual(3, self.env.last_col)
        self.assertEqual(2, self.env.last_row)

        self.assertEqual(
            ((0, 0), (1, 0), (2, 0), (3, 0),
             (0, 1), (2, 1), (3, 1), (0, 2),
             (1, 2), (2, 2), (3, 2)),
            self.env.states
        )

    def test_move_right(self):
        self.assertEqual((1, 0), self.env.attempt_move((0, 0), RIGHT))

    def test_move_right_to_wall(self):
        self.assertEqual((3, 0), self.env.attempt_move((3, 0), RIGHT))

    def test_move_left(self):
        self.assertEqual((0, 0), self.env.attempt_move((1, 0), LEFT))

    def test_move_left_to_wall(self):
        self.assertEqual((0, 0), self.env.attempt_move((0, 0), LEFT))

    def test_move_up(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 0), UP))

    def test_move_up_to_wall(self):
        self.assertEqual((0, 2), self.env.attempt_move((0, 2), UP))

    def test_move_down(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 2), DOWN))

    def test_move_down_to_wall(self):
        self.assertEqual((0, 0), self.env.attempt_move((0, 0), DOWN))

    def test_move_to_obstacle(self):
        self.assertEqual((0, 1), self.env.attempt_move((0, 1), RIGHT))

    def test_reset(self):
        init_states: Set[Tuple[int, int]] = set()
        for _ in range(10):
            init_states.add(self.env.reset())

        self.assertGreater(len(init_states), 3)

    def test_step(self):
        found_reward_state = False
        found_non_reward_state = False

        for _ in range(10):
            next_state, reward, done = self.env.step((2, 2), RIGHT)
            if next_state == (3, 2):
                self.assertEqual(1, reward)
                self.assertTrue(done)
                found_reward_state = True
            else:
                self.assertEqual(0, reward)
                self.assertFalse(done)
                found_non_reward_state = True

        self.assertTrue(found_reward_state)
        self.assertTrue(found_non_reward_state)
