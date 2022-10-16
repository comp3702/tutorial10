import argparse
import math
import random
import gym
import yaml

import numpy as np
from numpy import ndarray

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", default="Taxi-v3", help="Full name of the environment, e.g. Taxi-v3, FrozenLake-v1, etc.")
parser.add_argument("-c", "--config_file", default="config/q-learning.yaml", help="Config file with hyper-parameters")
args = parser.parse_args()

# Hyperparameters for the requried environment
hypers = yaml.load(open(args.config_file), Loader=yaml.FullLoader)

if args.env not in hypers:
    raise Exception(f'Hyper-parameters not found for env {args.env} - please add it to the config file (config/q-learning.yaml)')
params = hypers[args.env]

env = gym.make(args.env)

# Q table - a table of states x actions -> Q value for each possible action in each state
# should be initialized randomly !!! except for Q[terminal] !!! - but we don't know which one is terminal
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state: int, epsilon: float):
    if random.uniform(0, 1) < epsilon:
        # explore - i.e. choose a random action
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])


frame_idx = 0
rewards = []
episode_no = 0
max_r100 = -math.inf
epsilon = params['epsilon_start']

while True:
    state, _ = env.reset()
    action = choose_action(state, epsilon)

    episode_reward = 0
    done = False
    episode_start = frame_idx
    reward = 0

    while not done:
        epsilon = params['epsilon_final'] + (params['epsilon_start'] - params['epsilon_final']) * math.exp(-1.0 * frame_idx / params['epsilon_decay'])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frame_idx += 1
        episode_reward += reward

        # ===== update value table =====
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_error)
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (TD_target - Q_old(s, a))
        # Q_new(s,a) <-- Q_old(s,a) + alpha * (R + gamma*Q(s',a') - Q_old(s, a))
        # S' == next_state, a' == next_action
        next_action = choose_action(next_state, epsilon)
        Q_old = q_table[state, action]
        Q_next_old = q_table[next_state, next_action]

        Q_new = Q_old + params['alpha'] * (reward + params['gamma'] * Q_next_old - Q_old)

        q_table[state, action] = Q_new

        state = next_state
        action = next_action

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


