import datetime
import os
from eval import run
import submitit



#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

results_folder = os.path.join('results', 'our_update')

#connect to results folder
results_run = os.path.join(results_folder, '202307-1211-0003')
os.makedirs(results_run, exist_ok = True)

#make data folder inside timestamp
data = os.path.join(results_run, 'data')
os.makedirs(data, exist_ok = True)

#os.path.join(results, unique identifier)
training_path = os.path.join(data, "training_data")
os.makedirs(training_path, exist_ok = True)

evaluation_path = os.path.join(data, "evaluation_data")
os.makedirs(evaluation_path, exist_ok = True)

true_evaluation_path = os.path.join(evaluation_path, run_timestamp)
os.makedirs(true_evaluation_path, exist_ok=True)

outputpath = os.path.join(true_evaluation_path, "outputs")
os.makedirs(outputpath, exist_ok = True)

executor = submitit.AutoExecutor(folder=outputpath)
executor.update_parameters(timeout_min = 6000, mem_gb = 5, gpus_per_node = 0, cpus_per_task = 1, slurm_array_parallelism = 128)

jobs = []

penalties = [0, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64]
#penalties = [0]
train_healths = [1,2,3,4]
eval_healths = [1]
eta = 0.1
eps = 1000000

with executor.batch():
	for penalty in penalties:
		job = executor.submit(run, training_path=training_path, evaluation_path=true_evaluation_path, train_healths=train_healths, eval_healths=eval_healths, eta=eta, penalty=penalty, alpha=1, beta=1e-6, max_episodes=eps, Nagents=10, eval_episodes=1000)
		jobs.append(job)
