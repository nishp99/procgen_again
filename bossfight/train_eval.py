from training import train
from evalutation import evaluate
import os

def run(training_path, evaluation_path, penalty=0, alpha=1, beta=1e-6, max_episodes=500000, Nagents=10, eval_episodes=1000):
	train_path = os.path.join(training_path, f'pen{penalty}')
	#os.makedirs(train_path, exist_ok=True)
	eval_path = os.path.join(evaluation_path, f'pen{penalty}')
	#os.makedirs(evaluation_path, exist_ok=True)

	train(train_path, penalty, alpha, max_episodes, Nagents)
	evaluate(train_path, eval_path, eval_episodes, beta)

	return None



