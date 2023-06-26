import datetime
import os
from train_eval import run
import submitit



#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

results_folder = os.path.join('results', 'our_update')

#connect to results folder
results_run = os.path.join(results_folder, run_timestamp)
os.makedirs(results_run, exist_ok = True)

#make data folder inside timestamp
data = os.path.join(results_run, 'data')
os.makedirs(data, exist_ok = True)

#os.path.join(results, unique identifier)
training_path = os.path.join(data, "training_data")
os.makedirs(training_path, exist_ok = True)

evaluation_path = os.path.join(data, "evaluation_data")
os.makedirs(evaluation_path, exist_ok = True)

outputpath = os.path.join(results_run, "outputs")
os.makedirs(experiment_path, exist_ok = True)

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min = 6000, mem_gb = 5, gpus_per_node = 0, cpus_per_task = 1, slurm_array_parallelism = 128)

jobs = []

#penalties = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
penalties = [0]

with executor.batch():
	for penalty in penalties:
		job = executor.submit(run, training_path=training_path, evaluation_path=evaluation_path, penalty=penalty, alpha=1, beta=1e-6, max_episodes=5, Nagents=10, eval_episodes=5)
		jobs.append(job)