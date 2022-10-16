# Tutorial 10 - Q-Learning and SARSA

This is a significant departure from the official solutions:

- the GridWorld is modeled as episodic task
- removed EXIT_STATE - the episode ends when either of the reward states is reached
- the solution code is formulated as reinforcement learning problem, tracking progress through episodes and R100 (average reward of the past 100 episodes)
- unlike the official solution, consistently finds the same policy
- includes sample hyper-parameters for solving basic OpenAI Gym problems (Taxi-v3, FrozenLake-v1, FrozenLake8x8-v1)
- __provides ideas for hyper-parameter tuning__ - see at the bottom of this doc

### Dependencies
OpenAI Gym, PyYaml and Numpy

    pip install gym numpy pyyaml

#### Gym 0.26.2
There was a change in the OpenAI Gym's API recently and the step method now returns `next_state, reward, terminated, truncated, info`.
This used to be only `next_state, reward, done, info` - `terminated` and `truncated` used to be combined into `done`.

As such make sure you have at least version `0.26.2` or change the lines where `step()` and `reset()` are called with the old API.

### Running the Solutions

#### SimpleGridWorld

Start with [q_learning_simple_grid_world.py](q_learning_simple_grid_world.py) - this is the solution for the tutorial question.
The hyper-parameters are hardcoded to make it easier to follow.

    python q_learning_simple_grid_world.py

#### [OpenAI Gym](https://www.gymlibrary.dev/) Environments

[q_learning_gym.py](q_learning_gym.py) has the same code but the hyper-params are moved to the [config file](config/q-learning.yaml) for easier experimentation.
The sample hyper-parameters can solve both [Taxi](https://www.gymlibrary.dev/environments/toy_text/taxi/) and [FrozenLake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) every time (it did it in my tests) and FrozenLake8x8 most of the times.
You can experiment to see how the performance changes.

    python q_learning_gym.py -e Taxi-v3
    python q_learning_gym.py -e FrozenLake-v1
    python q_learning_gym.py -e FrozenLake8x8-v1

### SARSA
[sarsa_gym.py](sarsa_gym.py) has a basic implementation of SARSA algorithm.

    python sarsa_gym.py -e Taxi-v3
    python sarsa_gym.py -e FrozenLake-v1

## Hyper-parameter Tuning
You will need to tune the hyper-parameters for Assignment 3.

### Progress Indicators
While tuning, you need something to provide feedback throughout the training (not only at the end of the training).
For discrete problems, R100 (mean reward over past 100 episodes) and L100 (mean loss over past 100 steps) is used.

You want L100 to be decreasing and the R100 to be increasing.
It is also common to use R100 threshold for training termination. While you want this to be high, pushing it too high may lead to brittle policies that will not perform as well during non-exploration evaluation.

For Assignment 3, it may be useful to utilize `environment.evaluation_reward_tgt`. It may be good to target a bit lower value than this (e.g. x0.8), however, because the the problem is essentially a random function, going too low may actually produce a worse policy.

### Tunable Parameters
For Q-Learning/SARSA there are 3 parameters:
- alpha
- gamma
- epsilon (epsilon start, finish, and decay)

#### Alpha
Alpha controls the step size or how fast the training will be converging.
Setting it higher (e.g. 0.1 - the actual value depends on the environment) usually increases the initial progress, but may "overshoot" the optimal solution.
Setting it lower (e.g. 0.001) usually leads to very slow training that may get stuck in the initial exploration, but if you see a steady progress (increasing R100 and decreasing L100), it should usually converge.

Experiment with different values and watch the R100.

#### Epsilon
Some environments require more exploration and/or exploration over longer periods of training (e.g. as the agent gets further, it faces new/different challenges).
This is controlled by the decay rate / decay algorithm as well as starting/ending epsilon.

Epsilon settings also depend on the problem - if it is something learnable (like cartpole, pendulum, etc.) or a stochastic function (i.e. Assignment 3) that cannot be fully learned.

The best approach is to experiment a little and watch R100 / L100.

#### Gamma
Gamma is a discount factor and it is usually set quite high around 0.9. It affects the speed of convergence but I have rarely done much experimentation with it.