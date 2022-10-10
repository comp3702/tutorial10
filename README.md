# Tutorial 10 - Q-Learning

This is a significant departure from the official solutions:

- the GridWorld is modeled as episodic task
- removed EXIT_STATE - the episode ends when either of the reward states is reached
- the solution code is formulated as reinforcement learning problem, tracking progress through episodes and R100 (average reward of the past 100 episodes)
- unlike the official solution, consistently finds the same policy
- includes sample hyper-parameters for solving basic OpenAI Gym problems (Taxi-v3, FrozenLake-v1, FrozenLake8x8-v1)

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