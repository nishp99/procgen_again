from training import train
from evaluation import evaluate
import os

def run(data_path, weights_path, agent_health, policy, penalty=0, max_episodes=500000, num_save_points=110, Nagents=10, eval_agents=1000, lr=2e-4, discount=0.995, beta=1e-6):
	model_path = os.path.join(weights_path, 'weights')
	train(model_path, agent_health, policy, penalty, max_episodes, num_save_points, Nagents, lr, discount, beta)
	evaluate(data_path, model_path, policy, max_episodes, num_save_points, eval_agents)

	return None



