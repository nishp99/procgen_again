import gym3
from gym3 import types_np
import numpy as np
import os
#directory = path.path('evaluation.py').abspath()
#sys.path.append(directory.parent.parent)
import procgen.ProcgenGym3Env
#from procgen import ProcgenGym3Env

def train(file_path, penalty=0, alpha=1, max_episodes=500000, Nagents=10):
	agent_healths = np.array([1, 2, 3, 4])  # Training agent healths to use
	Nhealths = agent_healths.shape[0]

	save_points = np.unique(np.round(np.logspace(0,np.log10(max_episodes),110))) # A vector of episodes to save the weights at
	Nints = save_points.shape[0]
	save_ind = np.ones(Nagents)

	T=100
	# Maximum episode length. N.B. this is currently hard-coded in the C++ code and cannot be changed by changing this constant
	Nfeats = 6720 # Input feature dimension

	#env = ProcgenGym3Env(num=Nagents, env_name="bossfight", agent_health=5, use_backgrounds=False, restrict_themes=True)
	env = gym.make(num=Nagents, env_name="bossfight", agent_health=5, use_backgrounds=False, restrict_themes=True)
	# N.B. the agent_health argument is irrelevant--we do not use the returns computed by the environment/cpp code

	w = np.zeros((Nagents, Nhealths, Nfeats))
	ws = np.zeros((Nagents, Nints+1, Nhealths, Nfeats))
	y = np.zeros((Nagents, T+2))
	X = np.zeros((Nagents, Nfeats, T+2))
	acts = np.zeros(Nagents)
	a = np.zeros(Nagents)
	step = np.zeros(Nagents)

	total_episodes = np.zeros(Nagents)
	successful_episodes = np.zeros((Nagents, Nhealths))
	cumulative_rew = np.zeros(Nagents)

	while any(total_episodes <= max_episodes):
		rew, obs, first = env.observe()
		cumulative_rew += rew

		for i in range(Nagents):
			if step[i] > 0 and first[i]:  # First step of new episode
				step[i] = 0

				total_episodes[i] += 1

				successful_episode = cumulative_rew[i] > -agent_healths  # Vectorized for all agent healths

				successful_episodes[i, :] += successful_episode
				rewards = successful_episode.astype(int)
				rewards[rewards == 0] = -penalty

				# REINFORCE update
				u = np.mean(y[i, :] * X[i, :, :], axis=1).T
				w[i, :, :] = w[i, :, :] + alpha * np.outer(rewards, u)  # Vectorized for all agent healths

				cumulative_rew[i] = 0

				if any(save_points == total_episodes[i]):
					ws[i, save_ind[i].astype(int), :, :] = w[i, :, :]
					save_ind[i] += 1
					if i == 0:
						print(f"Saved episode {total_episodes[i]}")

				if i == 0 and total_episodes[i] % 1000 == 0:
					print(f"Iteration {total_episodes[i]}")

			x = obs['rgb'][i, 0:35, :, :].flatten()
			X[i, :, step[0].astype(int)] = x
			a[i] = np.random.rand() - 1 / 2  # Pure random policy

			if a[i] > 0:
				acts[i] = 0  # Left
				y[i, step[i].astype(int)] = 1
			else:

				acts[i] = 7  # Right
				y[i, step[i].astype(int)] = -1

			step[i] += 1

		env.act(acts)  # Take actions in all envs
	np.savez(file_path,ws=ws, Nagents=Nagents, Nints=Nints, Nfeats=Nfeats, agent_healths=agent_healths, save_points=save_points)