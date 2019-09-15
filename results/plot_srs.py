#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib
import sys
import os.path

import numpy as np

from conban_spanet.conbanalg import LAMB_DEFAULT
from bite_selection_package.config import spanet_config as config
N_FEATURES = 2048 if config.n_features==None else config.n_features


ESR_FULL = 0.99
ESR_PART = 0.137
PI_STAR = 0.603


def main():
	print(LAMB_DEFAULT, N_FEATURES)
	font = {'family' : 'sans',
        'size'   : 25}

	matplotlib.rc('font', **font)
	# Check argument
	# if len(file_lst) < 2:
	# 	print("Usage: plot_srs.py [file 1] [file 2] ...")

	plt.figure(1)
	out_csv = []

	file_lst = ["results/expected_srs_singleUCB_l{}_f{}_wo_banana.npz".format(LAMB_DEFAULT,N_FEATURES),
				"results/expected_srs_greedy_l{}_f{}_wo_banana.npz".format(LAMB_DEFAULT,N_FEATURES),
				"results/expected_srs_epsilon_0.1_l{}_f{}_wo_banana.npz".format(LAMB_DEFAULT,N_FEATURES), 
				"results/expected_srs_epsilon_0.3_l{}_f{}_wo_banana.npz".format(LAMB_DEFAULT,N_FEATURES),
				"results/expected_srs_epsilon_0.5_l{}_f{}_wo_banana.npz".format(LAMB_DEFAULT,N_FEATURES)]
	#while False:
	for i in range(len(file_lst)):
		# Check if file exists
		if not os.path.isfile(file_lst[i]):
			print("Error: NPZ file not found. (%s)" % file_lst[i])
			continue

		# Load NPZ into numpy array
		data = np.load(file_lst[i])["srs"]
		# Construct plotting data
		x = 10 * np.arange(len(data))

		# Plot
		if i == 0:
			conf = 1.96 * np.sqrt(data * (1.0 - data) / 47)
			plt.plot(x, data, color="r", label="LinUCB", linewidth=6.0)
			plt.fill_between(x, data-conf, data+conf, color='r', alpha=0.2)
		elif i==1:
			plt.plot(x, data, color="tab:gray", label="Greedy", linewidth=4.0)
		elif i == 2:
			plt.plot(x, data, color="g", label="epslion=0.1", linewidth=4.0)
		elif i == 3:
			plt.plot(x, data, color="y", label="epslion=0.3", linewidth=4.0)
		elif i == 4:
			plt.plot(x, data, color="b", label="epslion=0.5", linewidth=4.0)
		out_csv.append(data.reshape((34,1)))
	#plt.hlines(ESR_FULL, 0, 5000, linestyles='dashed', label="Trained SPANet")
	print("Writing CSV...")
	output_numpy = np.hstack(out_csv)
	print(output_numpy.shape)
	assert output_numpy.shape == (34,5)
	data_path = "./simu_data/"
	np.savetxt(os.path.join(data_path,"d{}_l{}.csv".format(N_FEATURES,LAMB_DEFAULT)), output_numpy, delimiter=",")
	plt.hlines(PI_STAR, 0, 5000, color='r', linestyles='dashed', label="Optimal", linewidth=4.0)
	plt.hlines(ESR_PART, 0, 5000, label="Baseline", linewidth=4.0)

	plt.xlabel("Number of Iterations")
	plt.ylabel("Expected Validation Success Rate")
	plt.xlim(0, 330)
	plt.xticks([0,100,200,300])
	plt.title("Test item: banana. d={}, lambda={}".format(N_FEATURES,LAMB_DEFAULT))
	#plt.xticks([0, 1000, 2000, 3000, 4000, 4500])
	plt.ylim(0, 0.9)
	plt.legend(bbox_to_anchor=(0., 0.82, 1., .082), loc=3,
       ncol=4, mode="expand")
	# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
 #       ncol=4, mode="expand")
	plt.show()
	plt.savefig("plot_d{}_l{}.jpg".format(N_FEATURES, LAMB_DEFAULT))





if __name__ == '__main__':
	main()
