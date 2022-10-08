import math
import random
import gym

import numpy as np

env = gym.make('Taxi-v3')

# Q table - a table of states x actions -> Q value for each possible action in each state
q_table = np.random.rand(env.observation_space.n, env.action_space.n)
# q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10000

max_episodes = 2000
frame_idx = 0
max_steps_per_episode = 100000
rewards = []

for episode_no in range(max_episodes):
    state, _ = env.reset()

    episode_reward = 0
    done = False
    episode_start = frame_idx
    reward = 0

    # print(q_table)

    while not done and (frame_idx - episode_start < max_steps_per_episode):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1.0 * frame_idx / epsilon_decay)
        if random.uniform(0, 1) < epsilon:
            # explore - i.e. choose a random action
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame_idx += 1
        episode_reward += reward

        # ===== update value table =====
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_error)
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_target - Q_old(s, a))
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (R + max_a(Q(s',a) - Q_old(s, a))
        # target = r + gamma * max_{a' in A} Q(s', a')
        Q_old = q_table[state, action]
        if done:
            Q_next_state_max = 0
        else:
            Q_next_state_max = np.max(q_table[next_state])

        Q_new = Q_old + alpha * (reward + gamma * Q_next_state_max - Q_old)

        q_table[state, action] = Q_new

        state = next_state

    rewards.append(reward)
    print(f"Episode {episode_no}, steps taken {frame_idx - episode_start}, reward: {episode_reward}, R100: {np.mean(rewards[-100:])}, epsilon: {epsilon}")

print(f"Steps taken {frame_idx}")
print(q_table)
policy = np.argmax(q_table, axis=1)


