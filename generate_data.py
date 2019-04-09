import os
import gym
import numpy as np
from IPython import embed
import pickle as pkl
from tqdm import tqdm

if not os.path.exists("data"):
    os.makedirs(directory)

env = gym.make('Pendulum-v0')
env.reset()
num_episodes = 10000
spe = 500
data = []

for episode in tqdm(range(num_episodes)):
	observation = env.reset()
	curr_episode = []
	for step in range(spe):
		state1 = observation
		action = np.array([4*np.random.rand() - 2])
		state2, reward, _, _ = env.step(action)
		state1 = np.append(state1, action[0])
		curr_episode.append([state1,state2])
		observation = state2
	data.append(curr_episode)

pkl.dump(data,open('data/gsimulation_data.pkl','wb'))
