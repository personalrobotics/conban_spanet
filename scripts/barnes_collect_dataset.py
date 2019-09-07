#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import csv
import warnings

from conban_spanet.spanet_driver import SPANetDriver
from bite_selection_package.config import spanet_config as config
import get_success_rate

N_FEATURES = 2048 if config.n_features==None else config.n_features
N_ACTIONS = 6



FOOD_NAME_TO_INDEX = {
	"_banana_": 0,
	"_honeydew_": 1,
	"_grape_": 2,
	"_spinach_": 3,
	"_cauliflower_": 4,
	"_strawberry_": 5,
	"_broccoli_": 6,
	"_kiwi_": 7,
	"_cherry_tomato_": 8,
	"_kale_": 9,
	"_lettuce_": 10,
	"_celery_": 11,
	"_bell_pepper_": 12,
	"_carrot_": 13,
	"_cantaloupe_": 14,
	"_apple_": 15
}

BACKGROUND_TO_INDEX = {
	"isolated": 0,
	"wall": 1,
	"lettuce+": 2
}

def get_food_name():
	return FOOD_NAME_TO_INDEX


def get_background_name():
	return BACKGROUND_TO_INDEX


def get_action_idx(filename):
	ret = 0
	if "tilted_angled" in filename:
		ret = 4
	elif "tilted_vertical" in filename:
		ret = 2

	if "angle-90" in filename:
		ret += 1

	return ret


def get_food_item(filename):
	"""
	Return index of food item in the filename.
	:param filename: String
	:return index:   Index of the food item.
	"""
	return __filename_to_index(filename, FOOD_NAME_TO_INDEX)


def get_background(filename):
	"""
	Return index of background (isolated, lettuce, against the wall) in the filename.
	:param filename: String
	:return index:   Index of the background.
	"""
	return __filename_to_index(filename, BACKGROUND_TO_INDEX)


def __filename_to_index(filename, dictionary):
	"""
	Checks whether any dictionary key is in filename, and if so, returns the key. Raises an error if not exactly 1 key
	is present in filename.
	:param filename:	String to check presence of keys.
	:param dictionary:  Dict mapping strings to integer values.
	:return index:      Value from dictionary.
	"""
	index = None
	for key, value in dictionary.items():
		if key in filename:
			if index is None:
				index = value
			else:
				raise KeyError('Multiple keys for filename {}'.format(filename))
	if index is None:
		warnings.warn('No key found in filename {}'.format(filename))
	return index


if __name__ == '__main__':

	# Initialize SPANet
	print("Path to store: {}".format(get_success_rate.data_path))
	print("Initializing Driver...")
	if get_success_rate.feat_version == 'all':
		food_to_exc = "banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi"
	else:
		food_to_exc = config.excluded_item
	driver = SPANetDriver(food_to_exc, None, 1, False, True)
	#driver = SPANetDriver("banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi", None, 1, False, True)
	#driver = SPANetDriver("banana_honeydew", None, 1, False, True)

	# Pull annotated filenames
	print("Pulling test file names...")
	ann_filenames = driver.dataset.ann_filenames
	print(ann_filenames)

	# Pull CSV, loss of 1 corresponds to a failure
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

		# Extract metadata from filename
		action_idx = get_action_idx(filename)
		food_idx = get_food_item(filename)
		background_idx = get_background(filename)

		# Get Success
		key = "+".join(filename.split("+", 2)[:2])
		if key not in l_dict:
				print("Warning, not in dict: " + key)
				continue
		l = l_dict[key]  # 1 is a failure

		# Concatonate features
		output_row = np.hstack((features, np.array([[action_idx, l, food_idx, background_idx]])))
		assert (output_row.shape == (1, N_FEATURES+4)), "Bad Shape for output!"

		output_csv.append(output_row)

	# Write CSV
	print("Writing CSV...")
	output_numpy = np.vstack(output_csv)
	#print(output_csv.shape)
	#print(output_numpy.shape)
	assert (output_numpy.shape == (len(output_csv), N_FEATURES + 4)), "Bad output array size!"
	np.savetxt(os.path.join(get_success_rate.data_path,"barnes_partial_dataset_test_all.csv"), output_numpy, delimiter=",")


