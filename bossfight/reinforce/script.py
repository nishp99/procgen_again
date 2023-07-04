import datetime
import os
from training_reinf import train
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

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min=6000, mem_gb=5, gpus_per_node=1, cpus_per_task=1, slurm_array_parallelism=40, slurm_partition="gpu")

jobs = []

#penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
agent_healths = [1]
penalties = [0]
discount = 0.995
beta = 0.01
Nagents = 10
max_episodes = 500000
lr = 2e-4

goods_path = os.path.join(results_run, f'agents{Nagents}eps{max_episodes}gamma{discount}beta{beta}')
os.makedirs(goods_path, exist_ok=True)

with executor.batch():
	for agent_health in agent_healths:
		for penalty in penalties:
			data_path = os.path.join(goods_path, f'health{agent_health}pen{penalty}')
			os.makedirs(data_path, exist_ok=True)
			job = executor.submit(train, data_path = data_path, agent_health=agent_health, penalty=penalty, max_episodes=max_episodes, Nagents=Nagents, lr=lr, discount=discount, beta=beta)
			#jobs.append(job)