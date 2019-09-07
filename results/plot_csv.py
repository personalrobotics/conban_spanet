#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib
import sys
import os.path

import numpy as np

NUM_ACTIONS = 6

def main():
	font = {'family' : 'sans',
		'size'   : 25}
	matplotlib.rc('font', **font)
	# Check argument
	if len(sys.argv) < 2:
		print("Usage: plot_csv.py [file 1] [file 2] ...")

	for i in range(1, len(sys.argv)):
		# Check if file exists
		if not os.path.isfile(sys.argv[i]):
			print("Error: CSV file not found. (%s)" % sys.argv[i])
			continue

		# Load CSV into numpy array
		with open(sys.argv[i], "rb") as csv_file:
			data = np.loadtxt(csv_file, delimiter=",")

			# Construct plotting data
			horizon = data.shape[0]
			timestep = np.arange(horizon)
			regret = np.cumsum(data[:, 0] - data[:, 1])

			actions = np.arange(NUM_ACTIONS)
			pi_star = np.bincount(data[:, 2].astype("int"))
			pi_unpad = np.bincount(data[:, 3].astype("int"))
			pi = np.zeros(actions.shape)
			pi[:pi_unpad.shape[0]] = pi_unpad

			# Plot
			plt.figure(1)
			if i == 1:
				plt.plot(timestep, regret, label="Greedy")
			else:
				plt.plot(timestep, regret, label="LinUCB")
			#plt.title("Regret")
			plt.xlabel("Time Step")
			plt.ylabel("Cumulative Regret")
			plt.xlim(xmin=0)
			plt.ylim(ymin=0)
			plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand")
			plt.legend()

			"""
			plt.figure(i + 1)
			width = 0.35
			plt.title("Action Choices: " + sys.argv[i])
			plt.xlabel("Action Index")
			plt.ylabel("# Uses")
			plt.bar(actions, pi_star, width, label="Pi Star")
			plt.bar(actions + width, pi, width, label="Pi")
			plt.xticks(actions + width / 2, ('VS0', 'VS90', 'TV0', 'TV90', 'TA0', 'TA90'))
			plt.legend()
			"""

	plt.show()






if __name__ == '__main__':
	main()
