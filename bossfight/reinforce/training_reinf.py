import gym3
from gym3 import types_np
import numpy as np
import torch
import torch.optim as optim
import os
import sys
import utils


current = os.path.dirname(os.path.realpath('training.py'))
parent = os.path.join(current, '..')
sys.path.append(os.path.join(parent, '..'))
#import procgen.ProcgenGym3Env
from procgen import ProcgenGym3Env

def train(data_path, agent_health, penalty=0, max_episodes=500000, Nagents=10, lr=2e-4, discount=0.995, beta=0.01): #,agent_health as additional parameter
	file_path = os.path.join(data_path, f'data_dic.npy')

	model_path = os.path.join(data_path, f'model.pt')

	RIGHT = 7
	LEFT = 0

	device = utils.device
	policy = utils.Policy().to(device)
	optimizer = optim.Adam(policy.parameters(), lr)

	"""save_points = np.unique(np.round(np.logspace(0,np.log10(max_episodes),110))) # A vector of episodes to save the weights at
	Nints = save_points.shape[0]
	save_ind = np.ones(Nagents)"""

	T=100
	episode_length = int(T/2 + 1)
	# Maximum episode length. N.B. this is currently hard-coded in the C++ code and cannot be changed by changing this constant

	#this implementation of vectorised environment could have errors later on in code due to differences to parallel_env.py
	env = ProcgenGym3Env(num=Nagents, env_name="bossfight", agent_health=5, use_backgrounds=False, restrict_themes=True)
	# N.B. the agent_health argument is irrelevant--we do not use the returns computed by the environment/cpp code

	step = 0
	total_episodes = 0

	dic = dict()
	dic['training'] = np.zeros((max_episodes, Nagents))
	dic['generalisation'] = np.zeros((max_episodes, Nagents))

	rews = np.zeros((episode_length, Nagents))
	cumulative_rew = np.zeros(Nagents)

	state_list = []
	prob_list = []
	action_list = []


	while total_episodes <= max_episodes:
		#change to observe, take a do nothing step, then observe to reduce number of total observations within episode
		#can change this after, lets see how it works out first
		rew, obs, done = env.observe()
		cumulative_rew += rew
		if any(done):
			#done = done_1
			#obs_1, obs_2 = obs_2, obs_1
			pass
		else:
			env.act([4]*Nagents) #do nothing action in all environments
			rew_2, obs, done = env.observe()
			cumulative_rew += rew_2
			rew += rew_2
			#done = done_2

		rews[step, :] = np.copy(rew)

		#if episode complete
		if step > 0 and np.any(done):
			successful_episode = cumulative_rew > -agent_health
			generalisation_success = cumulative_rew > -1  #we can do generalisation performance at the same time as training!

			dic['training'][total_episodes, :] = successful_episode.astype(int)
			dic['generalisation'][total_episodes, :] = generalisation_success.astype(int)

			#rewards calculated from successful/unsuccessful episodes
			end_rewards = successful_episode.astype(int)

			"""
			go through rews, using end_rewards as a mask check for failed episodes, find their indices
			loop through indices, for each failed episode find nth (Nagent'th) instance of non-zero element
			set all elements before this and after this to zero
			set the nth element to penalty
			loop through successful episodes
			set all elements to zero except final element set to 1
			"""
			fail_indices = np.where(end_rewards == 0)[0]
			success_indices = np.where(end_rewards != 0)[0]

			for i in range(np.shape(fail_indices)[0]):
				count = 1
				for j in range(episode_length):
					if rews[j, fail_indices[i]] < 0:
						if count == agent_health:
							rews[:, fail_indices[i]] = np.zeros(episode_length)
							rews[j, fail_indices[i]] = penalty
							break
						else:
							count += 1

			for i in range(np.shape(success_indices)[0]):
				rews[:, success_indices[i]] = np.zeros(episode_length)
				rews[-1, success_indices[i]] = 1


			"""
			backward on loss
			"""
			L = -utils.surrogate(policy, prob_list, state_list, action_list, rews, discount, beta)
			optimizer.zero_grad()
			L.backward()
			optimizer.step()
			del L

			total_episodes += 1
			#reset all values
			step = 0
			cumulative_rew = np.zeros(Nagents)
			beta *= 0.995 #reduces exploration in later runs

			#not necessary to save the weights if we calculate generalisation performance whilst training
			"""if any(save_points == total_episodes):
				true_model_path = os.path.join(model_path, f'model{total_episodes}.pt')
				torch.save(policy.state_dict(), true_model_path)
				print(f"Saved episode {total_episodes}")"""

			if total_episodes % 10000 == 0:
				torch.save(policy.state_dict(), model_path)
				np.save(file_path, dic)
				print(f"Iteration {total_episodes}")


		#require function to clean and batch frame, ready for input to network
		#need to turn into torch tensor and make channels the 2nd axis
		#then centre images (take away per channel mean values) and divide by 255?
		# (Nagents, 64, 64, 3)
		batch_input = utils.preprocess_batch(obs['rgb'])
		# probs will only be used as the pi_old
		# no gradient propagation is needed
		# so we move it to the cpu
		probs = policy(batch_input).squeeze().cpu().detach().numpy()
		acts = np.where(np.random.rand(Nagents) < probs, RIGHT, LEFT)
		probs = np.where(acts == RIGHT, probs, 1.0 - probs)

		#store the results
		state_list.append(batch_input)
		prob_list.append(probs)
		action_list.append(acts)

		step += 1

		env.act(acts)  # Take actions in all envs
	return None