import gym3
from gym3 import types_np
import numpy as np
import sys
import path
#directory = path.path('evaluation.py').abspath()
#sys.path.append(directory.parent.parent)

#from procgen_fork.procgen import ProcgenGym3Env
import procgen.ProcgenGym3Env

# Evaluate agents using softmax policy

def evaluate(file_path, save_path,eval_episodes=1000, alpha=1e-6):

	data = np.load(file_path, allow_pickle = True)
	ws = data['arr_0']
	Nagents = data['arr_1']
	Nints = data['arr_2']
	Nfeats = data['arr_3']
	agent_healths = data['arr_4']
	Nhealths = agent_healths.shape[0]
	save_points = data['arr_5']

	env = ProcgenGym3Env(num=Nagents, env_name="bossfight", agent_health=5, use_backgrounds=False, restrict_themes=True)

	successful_episodes = np.zeros((Nagents, Nints + 1, Nhealths, Nhealths))  # first Nhealth is train, second is test

	max_episodes = Nints * eval_episodes

	for j, train_ah in enumerate(agent_healths):
		acts = np.zeros(Nagents)
		a = np.zeros(Nagents)
		step = np.zeros(Nagents)

		total_episodes = np.zeros(Nagents)

		cumulative_rew = np.zeros(Nagents)

		while any(total_episodes <= max_episodes):
			rew, obs, first = env.observe()
			cumulative_rew += rew

			for i in range(Nagents):
				if step[i] > 0 and first[i]:  # First step of new episode
					step[i] = 0

					total_episodes[i] += 1

					interval = int(total_episodes[i] / eval_episodes)

					successful_episode = cumulative_rew[i] > -agent_healths  # Vectorized for all agent healths

					successful_episodes[i, interval, j, :] += successful_episode

					cumulative_rew[i] = 0

					if i == 0 and total_episodes[i] % 1000 == 0:
						print(f"Iteration {total_episodes[i]}")

				x = obs['rgb'][i, :, :, :].flatten()

				z = x.reshape((64, 64, 3))
				z[28:35, 28:35, :] = 0  # Remove spacecraft (turns out to be essential, otherwise get strong side bias
				z = z[0:35, :]  # Remove bottom half because there's nothing there
				x = z.flatten()

				interval = int(total_episodes[i] / eval_episodes)
				s = 1. / (1 + np.exp(-alpha * ws[i, interval, j, :] @ x))
				a[i] = 2 * ((np.random.rand() < s) - 1 / 2)

				if a[i] > 0:
					acts[i] = 0  # Left
				else:

					acts[i] = 7  # Right

				step[i] += 1

			env.act(acts)  # Take actions in all envs

	np.savez(save_path, successful_episodes=successful_episodes,
			 eval_episodes=eval_episodes, agent_healths=agent_healths, save_points=save_points)