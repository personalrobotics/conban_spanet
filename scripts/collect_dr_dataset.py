#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import csv

from conban_spanet.spanet_driver import SPANetDriver
from bite_selection_package.config import spanet_config as config

N_FEATURES = 2048 if config.n_features==None else config.n_features
N_ACTIONS = 6

def get_action_idx(filename):
	ret = 0
	if "tilted_angled" in filename:
		ret = 4
	elif "tilted_vertical" in filename:
		ret = 2

	if "angle-90" in filename:
		ret += 1

	return ret

if __name__ == '__main__':

	# Initialize SPANet
	print("Initializing Driver...")
	#driver = SPANetDriver("banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi", None, 1, False, True)
	driver = SPANetDriver("banana_honeydew", None, 1, False, True)

	# Pull annotated filenames
	print("Pulling test file names...")
	ann_filenames = driver.dataset.ann_filenames

	# Pull CSV
	print("Reading loss CSV...")
	l_dict = {}
	with open('consolidated_successes.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			l_dict[row[0]] = float(row[1])

	# For Each File in Dataset
	print("Running on %d files..." % len(ann_filenames))
	output_csv = []
	for i in range(len(ann_filenames)):
		if i % 100 == 0:
			print("i: " + str(i))
		filename = ann_filenames[i]
		_, gv, features = driver._sample_dataset(i)

		# Get Action index
		action_idx = get_action_idx(filename)

		# Get Success
		key = "+".join(filename.split("+", 2)[:2])
		if key not in l_dict:
				print("Warning, not in dict: " + key)
				continue
		l = l_dict[key]  # 1 is failure

		# Calculate DR vector, note expected loss = (1-gv)
		l_hat = 1.0 - gv
		# Already taken into account in spanet_dataset.py
		#l_hat[0, action_idx] += (6.0 * (l - l_hat[0, action_idx]))

		# Concatonate features
		output_row = np.hstack((features, l_hat))
		assert (output_row.shape == (1, N_FEATURES+N_ACTIONS)), "Bad Shape for output!"

		output_csv.append(output_row)

	# Write CSV
	print("Writing CSV...")
	output_numpy = np.vstack(output_csv)
	#print(output_csv.shape)
	#print(output_numpy.shape)
	assert (output_numpy.shape == (len(output_csv), N_FEATURES + N_ACTIONS)), "Bad output array size!"
	np.savetxt("dr_dataset.csv", output_numpy, delimiter=",")


