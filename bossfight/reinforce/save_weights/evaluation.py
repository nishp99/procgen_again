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


def evaluate(data_path, weights_path, policy, max_episodes, num_save_points, eval_agents): #,agent_health as additional parameter
	full_file_path = os.path.join(data_path, 'dic.npy')

	RIGHT = 7
	LEFT = 0

	#device = utils.device
	device = 'cpu'

	save_points = np.unique(np.round(np.logspace(0, np.log10(max_episodes), num_save_points))) # A vector of episodes to save the weights at
	save_points = save_points.tolist()

	dic = dict()
	dic['generalisation'] = np.zeros(len(save_points))

	if policy == 'deep':
		policy = utils.Policy().to(device)
	elif policy == 'shallow':
		policy = utils.Shallow().to(device)
	elif policy == 'twolayer':
		policy = utils.Twolayer().to(device)

	for i, t in enumerate(save_points):
		save_index = int(t)
		true_model_path = os.path.join(weights_path, f'{save_index}.pt')
		policy.load_state_dict(torch.load(true_model_path))

		#instantiate environment
		env = ProcgenGym3Env(num=eval_agents, env_name="bossfight", agent_health=5, use_backgrounds=False, restrict_themes=True)
		# N.B. the agent_health argument is irrelevant--we do not use the returns computed by the environment/cpp code

		step = 0
		cumulative_rew = np.zeros(eval_agents)

		while True:
			#change to observe, take a do nothing step, then observe to reduce number of total observations within episode
			#can change this after, lets see how it works out first
			rew, obs, done = env.observe()
			cumulative_rew += rew
			if any(done):
				#done = done_1
				#obs_1, obs_2 = obs_2, obs_1
				pass
			else:
				env.act(4*np.zeros(eval_agents)) #do nothing action in all environments
				rew_2, obs, done = env.observe()
				cumulative_rew += rew_2
				rew += rew_2
				#done = done_2

			#rews[step, :] = np.copy(rew)

			#if episode complete
			if step > 0 and np.any(done):
				#successful_episode = cumulative_rew > -agent_health
				generalisation_success = cumulative_rew > -1  #we can do generalisation performance at the same time as training!

				#dic['training'][total_episodes, :] = successful_episode.astype(int)
				dic['generalisation'][i] = np.mean(generalisation_success.astype(int))

				#rewards calculated from successful/unsuccessful episodes
				#end_rewards = successful_episode.astype(int)

				#total_episodes += 1
				#reset all values
				step = 0
				cumulative_rew = np.zeros(eval_agents)

				#save data
				np.save(full_file_path, dic)
				print(f"Saved episode {t}")
				break


			#require function to clean and batch frame, ready for input to network
			#need to turn into torch tensor and make channels the 2nd axis
			#then centre images (take away per channel mean values) and divide by 255?
			# (Nagents, 64, 64, 3)
			batch_input = utils.preprocess_batch(obs['rgb'])
			probs = policy(batch_input).squeeze().detach().numpy()
			acts = np.where(np.random.rand(eval_agents) < probs, RIGHT, LEFT)

			step += 1

			env.act(acts)  # Take actions in all envs
	return None