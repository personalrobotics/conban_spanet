#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import csv

from conban_spanet.spanet_driver import SPANetDriver

if __name__ == '__main__':

	# Initialize SPANet
	print("Initializing Driver...")
	driver = SPANetDriver("banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi", None, 1, False, True)

	# Pull annotated filenames
	print("Pulling test file names...")
	ann_filenames = driver.dataset.ann_filenames

	# Pull CSV
	print("Reading loss CSV...")
	r_dict = {}
	with open('consolidated_successes.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			r_dict[row[0]] = float(row[1])

	# For Each File in Dataset
	for i in range(len(ann_filenames)):
		filename = ann_filenames[i]
		_, gv, features = driver._sample_dataset(i)

		# Get Action index
		action_idx = get_action_idx(filename)

		# Get Success
		key = "+".join(filename.split("+", 2)[:2])
		if key not in r_dict:
				print("Warning, not in dict: " + key)
				continue
		r = r_dict[key]
