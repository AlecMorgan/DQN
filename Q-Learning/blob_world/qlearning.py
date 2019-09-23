"""
A custom blob world environment for our 
algorithms to solve. The agent plays a 
blob on a small grid and must walk around
collecting food blobs and avoiding enemy
blobs.
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
import time

np.random.seed(0)

plt.style.use("ggplot")

# 10x10 grid
SIZE = 10
NUM_EPISODES = 25000
MOVE_PENALTY = 1 # penalty for moving
ENEMY_PENALTY = 300 # penalty for hitting an enemy
FOOD_REWARD = 25 # reward for obtaining food

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {
    1: (255, 175, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
    }

# Mapping of which actions corresponding to which movement tuples
movement = {
    0: ( 1,  1),
    1: (-1, -1),
    2: (-1,  1),
    3: ( 1, -1)
}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"Blob at coordinates x={self.x}, y={self.y}"

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def action(self, choice):
        # If action choice is a movement
        if choice in [0, 1, 2, 3]:
            x, y = movement[choice]
            self.move(x, y)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        
        if self.x < 0: 
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        
        if self.y < 0: 
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else: 
    with open(start_q_table, "rb") as f:
        q_table = pickle

episode_rewards = []
for episode in range(NUM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0:
        print(f"On #{episode}, with epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)

        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else: 
            reward = -MOVE_PENALTY

        # Observations are simply the relative coords of food and enemies
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        # Q value for our current state-action combination
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        # INDENTATION?
        if show: 
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else: 
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)
    epsilon*= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("Episode #")
plt.show()

# Save qtable
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)