#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib
import sys
import os.path

import numpy as np

LAMB_DEFAULT = 100
N_FEATURES = 2048


ESR_FULL = 0.99
ESR_PART = 0.010
PI_STAR = 0.176

def main():
	font = {'family' : 'sans',
        'size'   : 25}

	matplotlib.rc('font', **font)

	plt.figure(1)

	#while False:

	# Load NPZ into numpy array
	data = np.genfromtxt('EMA_data.csv', delimiter=',')

	# Construct plotting data

	x = data[:,0]
	EMA_LinUCB = data[:,1]
	EMA_Greedy = data[:,2]

	# Plot
	plt.plot(x, EMA_Greedy, color="tab:gray", label="Greedy", linewidth=4.0)
	#conf = 1.96 * np.sqrt(data * (1.0 - data) / 47)
	plt.plot(x, EMA_LinUCB, color="r", label="LinUCB", linewidth=6.0)
	#plt.fill_between(x, data-conf, data+conf, color='r', alpha=0.2)

	#plt.hlines(ESR_FULL, 0, 5000, linestyles='dashed', label="Trained SPANet")
	plt.hlines(PI_STAR, 0, 5000, color='r', linestyles='dashed', label="Optimal", linewidth=4.0)
	plt.hlines(ESR_PART, 0, 5000, label="Baseline", linewidth=4.0)

	plt.xlabel("Number of Iterations")
	plt.ylabel("Mean Success Rate")
	plt.xlim(0, 56)
	plt.xticks([0,28,55])
	plt.title("Test item: banana. d={}, lambda={}".format(N_FEATURES,LAMB_DEFAULT))
	#plt.xticks([0, 1000, 2000, 3000, 4000, 4500])
	plt.ylim(0, 0.75)
	plt.legend(bbox_to_anchor=(0., 0.82, 1., .082), loc=3,
       ncol=4, mode="expand")
	# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
 #       ncol=4, mode="expand")
	plt.savefig("plot_d{}_l{}.jpg".format(N_FEATURES, LAMB_DEFAULT))
	plt.show()






if __name__ == '__main__':
	main()
