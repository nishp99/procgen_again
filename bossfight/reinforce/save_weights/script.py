import datetime
import os
from train_eval import run
import submitit



#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)


#connect to results folder
results_run = os.path.join('results', run_timestamp)
os.makedirs(results_run, exist_ok=True)

#make data folder inside timestamp

"""#os.path.join(results, unique identifier)
training_path = os.path.join(data, "training_data")
os.makedirs(training_path, exist_ok = True)

evaluation_path = os.path.join(data, "evaluation_data")
os.makedirs(evaluation_path, exist_ok = True)"""

outputpath = os.path.join(results_run, "outputs")
os.makedirs(outputpath, exist_ok=True)

ceph_path = os.path.realpath(os.path.join(os.path.dirname('script.py'), '..', '..', '..', '..', '..', '..', '..', '..', 'ceph', 'saxe', 'npatel', run_timestamp))

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min=10000, mem_gb=5, gpus_per_node=1, cpus_per_task=28, slurm_array_parallelism=128, slurm_partition="gpu")

jobs = []

#penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
agent_healths = [1]
penalties = [0]
discount = 0.995
beta = 0.01
#num_save_points = 110
num_save_points = 2
#Nagents = 10
Nagents = 2
#max_episodes = 500000
max_episodes = 5
#eval_agents = 10
eval_agents = 2
lr = 2e-4
trials = 1
policy = 'shallow'

goods_path = os.path.join(results_run, f'agents{Nagents}eps{max_episodes}gamma{discount}beta{beta}')
os.makedirs(goods_path, exist_ok=True)

with executor.batch():
	for agent_health in agent_healths:
		for penalty in penalties:
			for trial in range(trials):
				data_path = os.path.join(goods_path, f'health{agent_health}pen{penalty}trial{trial}')
				weights_path = os.path.join(ceph_path, f'health{agent_health}pen{penalty}trial{trial}')
				os.makedirs(data_path, exist_ok=True)
				os.makedirs(weights_path, exist_ok=True)
				job = executor.submit(run, data_path=data_path, weights_path=weights_path, agent_health=agent_health, policy=policy, penalty=penalty, max_episodes=max_episodes, num_save_points=num_save_points, Nagents=Nagents, eval_agents=eval_agents, lr=lr, discount=discount, beta=beta)
				#jobs.append(job)