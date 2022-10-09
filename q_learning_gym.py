import argparse
import math
import random
import gym
import yaml

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="Taxi-v3", help="Full name of the environment, e.g. Taxi-v3, FrozenLake-v1, etc.")
parser.add_argument("-c", "--config_file", default="config/q-learning.yaml", help="Config file with hyper-parameters")
args = parser.parse_args()

env = gym.make(args.env)

# Q table - a table of states x actions -> Q value for each possible action in each state
q_table = np.random.rand(env.observation_space.n, env.action_space.n)
# q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters for the requried environment
params = yaml.load(open(args.config_file), Loader=yaml.FullLoader)[args.env]

frame_idx = 0
rewards = []
episode_no = 0
max_r100 = -math.inf

while True:
    state, _ = env.reset()

    episode_reward = 0
    done = False
    episode_start = frame_idx
    reward = 0

    while not done:
        epsilon = params['epsilon_final'] + (params['epsilon_start'] - params['epsilon_final']) * math.exp(-1.0 * frame_idx / params['epsilon_decay'])
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

        Q_new = Q_old + params['alpha'] * (reward + params['gamma'] * Q_next_state_max - Q_old)

        q_table[state, action] = Q_new

        state = next_state

    rewards.append(reward)
    r100 = np.mean(rewards[-100:])
    if r100 > max_r100:
        max_r100 = r100
    print(f"Frame: {frame_idx}, Episode {episode_no}, steps taken {frame_idx - episode_start}, reward: {episode_reward}, R100: {r100}, max R100: {max_r100}, epsilon: {epsilon}")

    if r100 >= params['stopping_reward']:
        print(f"Solved after {frame_idx} frames with R100 of {r100}")
        break

    if frame_idx > params['max_frames']:
        print(f"Ran out of time after {frame_idx} frames")
        break
    episode_no += 1

print(f"Steps taken {frame_idx}")
print(q_table)
policy = np.argmax(q_table, axis=1)


