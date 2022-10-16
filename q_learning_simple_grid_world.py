import math
import random

import numpy as np

from envs.SimpleGridWorld import SimpleGridWorld, ACTION_NAMES

env = SimpleGridWorld()

# Q table - a table of states x actions -> Q value for each possible action in each state
q_table = np.random.rand(len(env.states), len(env.actions))
# q_table = np.zeros((len(env.states), len(env.actions)))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 1000

max_episodes = 1000
frame_idx = 0
max_steps_per_episode = 1000
rewards = []

for episode_no in range(max_episodes):
    state = env.reset()

    episode_reward = 0
    done = False
    episode_start = frame_idx

    # print(q_table)

    while not done and (frame_idx - episode_start < max_steps_per_episode):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1.0 * frame_idx / epsilon_decay)
        state_index = env.states.index(state)

        if random.uniform(0, 1) < epsilon:
            # explore - i.e. choose a random action
            action = random.choice(env.actions)
        else:
            action = np.argmax(q_table[state_index])

        next_state, reward, done = env.step(state, action)
        episode_reward += reward
        frame_idx += 1

        # ===== update value table =====
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_error)
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_target - Q_old(s, a))
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (R + max_a(Q(s',a) - Q_old(s, a))
        # target = r + gamma * max_{a' in A} Q(s', a')
        Q_old = q_table[state_index, action]
        if done:
            Q_next_state_max = 0
        else:
            next_state_index = env.states.index(next_state)
            Q_next_state_max = np.max(q_table[next_state_index])

        Q_new = Q_old + alpha * (reward + gamma * Q_next_state_max - Q_old)

        q_table[state_index, action] = Q_new

        state = next_state

    rewards.append(episode_reward)
    print(f"Episode {episode_no}, steps taken {frame_idx - episode_start}, reward: {episode_reward}, R100: {np.mean(rewards[-100:])}, epsilon: {epsilon}")

print(f"Steps taken {frame_idx}")
print("Q-Table:")
print(q_table)
policy = np.argmax(q_table, axis=1)

for row in range(env.last_row, -1, -1):
    for col in range(env.last_col + 1):
        state = (col, row)
        if state in env.states:
            state_index = env.states.index(state)
            action = policy[state_index]
            print(ACTION_NAMES[action], end=' ')
        else:
            print('X', end=' ')
    print("")

