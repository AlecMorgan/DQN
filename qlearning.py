import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()

# How quickly should we change our behavior to make
# sure we don't over- or under-shoot our goals? 
LEARNING_RATE = 0.1 
# How valuable are future rewards? 
DISCOUNT = 0.95
# Training epochs.
EPISODES = 2000
# How many episodes to wait in between rendering a given episode. 
SHOW_EVERY = 500

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


# Epsilon-greediness is a measure of how to balance exploration vs exploitation. 
# For example, epsilon 0.05 would mean that we'll exploit the best thing we know 
# 95% of the time and try new, potentially better options 5% of the time. 
epsilon = 0.5
START_EPISLON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPISLON_DECAYING)

# Initialize Q-table with random values. 
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Three actions: 0 = move left, 2 = move right, 1 = do nothing. 
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render() # Rendering each episode significantly increases runtime.

        if not done:
            # Maximum future Q-value for a particular sequence of actions.
            # In other words, "How good or not was this path that we just took?"
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            # New Q-value to update Q-table with.
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        # If we reach our goal, "reward" = 0 instead of -1.
        elif new_state[0] >= env.goal_position:
            # print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    
    # Over time epsilon will decrease, meaning that we will exploit more than we will explore
    if END_EPSILON_DECAYING >= episode >= START_EPISLON_DECAYING:
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    if episode % 10 == 0:
        np.save(f"qtables/qtable-{episode}.npy", q_table)
    if episode % SHOW_EVERY == 0:
        last_k_ep_rewards = ep_rewards[-SHOW_EVERY:]
        average_reward = sum(last_k_ep_rewards) / len(last_k_ep_rewards)
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(last_k_ep_rewards))
        aggr_ep_rewards['max'].append(max(last_k_ep_rewards))
        
        print(f"""
        Episode: {episode} 
        Avg: {average_reward} 
        Min: {min(last_k_ep_rewards)} 
        Max: {max(last_k_ep_rewards)}
        """)

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()