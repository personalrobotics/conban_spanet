import subprocess
import os
import numpy as np

from bite_selection_package.config import spanet_config as config
N_FEATURES = 2048 if config.n_features==None else config.n_features

import rospkg
rospack = rospkg.RosPack()
data_path = os.path.join(rospack.get_path('conban_spanet'), "results/")


# lambd_list = [10.0, 100.0, 1000.0, 10000.0]
# for lambd in lambd_list:
# 	subprocess.run(["python3", "scripts/run_experiment.py","-a","greedy","-lbd",str(lambd)])

# file_lst = list( map(lambda l: "greedy_l_{}_f_{}_wo_banana_apple_grape.npz".format(l,N_FEATURES),lambd_list))
# file_lst = list(map(lambda s: os.path.join(data_path, s), file_lst))
# algo_loss_lst = list( map(lambda file: np.sum(np.load(file)["pi_loss"]), file_lst))


# print()
# print("d = {}, losses for these lambda {} are:".format(N_FEATURES, lambd_list))
# print("algo_loss_lst:  ", algo_loss_lst)
# print("The minimum loss is {}, which corresponds to {}".format(np.min(algo_loss_lst), lambd_list[np.argmin(algo_loss_lst)]))





lambd = 1000.0
eps_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
for eps in eps_list:
	subprocess.run(["python3", "-u", "scripts/run_experiment.py","-a","epsilon","-eps",str(eps),"-lbd",str(lambd)])
file_lst = list( map(lambda eps: "epsilon_e_{}_l_{}_f_{}_wo_banana_apple_grape.npz".format(eps, lambd,N_FEATURES),eps_list))
file_lst = list(map(lambda s: os.path.join(data_path, s), file_lst))
algo_loss_lst = list( map(lambda file: np.sum(np.load(file)["pi_loss"]), file_lst))
print()
print("d = 2048, losses for these epsilon {} are:".format(eps_list))
print("algo_loss_lst:  ", algo_loss_lst)
print("The minimum loss is {}, which corresponds to epsilon={}".format(np.min(algo_loss_lst), eps_list[np.argmin(algo_loss_lst)]))


alpha_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for alpha in alpha_list:
	subprocess.run(["python3", "-u", "scripts/run_experiment.py","-a","singleUCB", "-alp", str(alpha),"-lbd",str(lambd)])
file_lst = list( map(lambda alpha: "singleUCB_alpha_{}_gamma_1000_l_{}_f_{}_wo_banana_apple_grape.npz".format(alpha, lambd,N_FEATURES),alpha_list))
file_lst = list(map(lambda s: os.path.join(data_path, s), file_lst))
algo_loss_lst = list( map(lambda file: np.sum(np.load(file)["pi_loss"]), file_lst))
print()
print("d = 2048, losses for these alpha {} are:".format(alpha_list))
print("algo_loss_lst:  ", algo_loss_lst)
print("The minimum loss is {}, which corresponds to alpha={}".format(np.min(algo_loss_lst), alpha_list[np.argmin(algo_loss_lst)]))
