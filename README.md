# DQN
An implementation of the original deep reinforcement learning algorithm, DQN (Deep Q. This implementation was built solely for learning purposes by following Sentdex's excellent [tutorial series](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7). 

DQN (Deep Q-Learning) is a variant of Q-Learning which uses deep neural networks. Q-Learning refers to algorithms which learn an action-value function `Q(s, a)` which attempts to predict the Q or *quality* of a given state `s` which we will receive given that we take a certain action `a`. Deep Q-Learning simply uses deep neural networks to learn this function. 

DQN is an especially noteworthy algorithm as it was the first deep reinforcement learning algorithm ever created. It was introduced by DeepMind in their [2013 paper](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning) in which this single algorithm was demonstrated to achieve superhuman performance on a large number of Atari games. Today it has been superseded by a variety of other deep reinforcement learning algorithms such as PPO and A3C, but DQN is still the subject of active academic interest as evidenced by emerging variants such as Double DQN and Rainbow. 

From a learning perspective, DQN is especially interesting because for all its effectiveness and complexity, it's still one of the simplest deep reinforcement learning algorithms. It has been aptly called the "Hello World" of deep reinforcement learning. 
