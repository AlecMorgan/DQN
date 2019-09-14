import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

"""
DISCRETE_OS_SIZE = discrete observation space size.
Continuous environments with multiple variables quickly 
result in combinatorial explosions. This is highly problematic
for Q-learning since we intend to fill a table with all 
possible combinations. Approximating the full gamut of
possible values using discrete substitutes creates an
effective and tractable approximation. 
"""
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Initialize Q-table with random values. 
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# print(discrete_os_win_size)

# done = False

# while not done:
#     # Three actions: 0 = move left, 2 = move right, 1 = do nothing. 
#     action = 2
#     new_state, reward, done, _ = env.step(action)
#     print(new_state)
#     env.render()

# env.close()