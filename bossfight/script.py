import datetime
import os
from train_eval import run
import submitit

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
training_path = os.path.join(run_timestamp, "training_data")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "outputs")
os.makedirs(experiment_path, exist_ok = True)

evaluation_path = os.path.join(run_timestamp, "evaluation_data")
os.makedirs(results_path, exist_ok = True)

run_path = os.path.join('results', results_path)
os.mkdir(run_path)

executor = submitit.AutoExecutor(folder="results/our_update")

executor.update_parameters(timeout_min = 6000, mem_gb = 5, gpus_per_node = 0, cpus_per_task = 1, slurm_array_parallelism = 128)

jobs = []

#penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
penalties = [0]

with executor.batch():
	for penalty in penalties:
		job = executor.submit(run, training_path=training_path, evaluation_path=evaluation_path, penalty=penalty, alpha=1, beta=1e-6, max_episodes=5, Nagents=10, eval_episodes=5)
		jobs.append(job)