#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib
import sys
import os.path

import numpy as np

ESR_FULL = 0.99
ESR_PART = 0.137
PI_STAR = 0.552

def main():
	font = {'family' : 'sans',
        'size'   : 25}

	matplotlib.rc('font', **font)
	# Check argument
	if len(sys.argv) < 2:
		print("Usage: plot_srs.py [file 1] [file 2] ...")

	plt.figure(1)

	#while False:
	for i in range(1, len(sys.argv)):
		# Check if file exists
		if not os.path.isfile(sys.argv[i]):
			print("Error: NPZ file not found. (%s)" % sys.argv[i])
			continue

		# Load NPZ into numpy array
		data = np.load(sys.argv[i])["srs"]

		# Construct plotting data

		x = 10 * np.arange(len(data))

		# Plot
		if i == 1:
			plt.plot(x, data, color="tab:gray", label="Greedy", linewidth=4.0)
		else:
			conf = 1.96 * np.sqrt(data * (1.0 - data) / 47)
			plt.plot(x, data, color="r", label="LinUCB", linewidth=6.0)
			plt.fill_between(x, data-conf, data+conf, color='r', alpha=0.2)

	#plt.hlines(ESR_FULL, 0, 5000, linestyles='dashed', label="Trained SPANet")
	plt.hlines(PI_STAR, 0, 5000, color='r', linestyles='dashed', label="Optimal", linewidth=4.0)
	plt.hlines(ESR_PART, 0, 5000, label="Baseline", linewidth=4.0)

	plt.xlabel("Number of Iterations")
	plt.ylabel("Expected Validation Success Rate")
	plt.xlim(0, 330)
	plt.xticks([0,100,200,300])
	plt.title("Test item: banana. d=512, lambda=10")
	#plt.xticks([0, 1000, 2000, 3000, 4000, 4500])
	plt.ylim(0, 0.9)
	plt.legend(bbox_to_anchor=(0., 0.82, 1., .082), loc=3,
       ncol=4, mode="expand")
	# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
 #       ncol=4, mode="expand")
	plt.show()






if __name__ == '__main__':
	main()
