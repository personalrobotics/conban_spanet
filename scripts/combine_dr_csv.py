#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import csv
import torch

from conban_spanet.spanet_driver import SPANetDriver
from bite_selection_package.config import spanet_config as config


N_FEATURES = 2048 if config.n_features==None else config.n_features
N_ACTIONS = 6

if __name__ == '__main__':
	_dir = "/home/conban/dr_dataset"
	_exclude_food = '_wo_{}'.format(config.excluded_item) if config.excluded_item else ''
	# train_csv = os.path.join(_dir, "dr_dataset_dr_n"
	# 	+str(N_FEATURES)+"{}_train.csv".format(_exclude_food))
	# test_csv =  os.path.join(_dir, "dr_dataset_dr_n"
	# 	+str(N_FEATURES)+"{}_test.csv".format(_exclude_food))
	# print("Get train csv from: {}".format(train_csv))
	# print("Get test csv from: {}".format(test_csv))
	# train = np.genfromtxt(train_csv, delimiter=',')
	# test = np.genfromtxt(test_csv,  delimiter=",")

	# #_npz = os.path.join(_dir, "dr_dataset_n"+str(N_FEATURES)+".npz")
	# _npz = "dr_dataset_n"+str(N_FEATURES)+"{}.npz".format(_exclude_food)
	# print("Save dr npz as: {}".format(_npz))
	# np.savez_compressed(_npz, train=train, test=test)
	
	_feat_str = "_n"+str(N_FEATURES) if config.n_features!=None else ''
	ckpt_path = "/home/conban/conban_ws/src/bite_selection_package/checkpoint/food_spanet_all"+_feat_str+"_rgb_wall"+_exclude_food+"_dr_ckpt_best.pth"
	checkpoint = torch.load(ckpt_path) 
	print("Get checkpoint from: {}".format(ckpt_path))
	weight = checkpoint['net']['final.weight'].cpu().numpy() 
	bias = checkpoint['net']['final.bias'].cpu().numpy()
	_npz1 = "spanet_untrained_dr_n"+str(N_FEATURES)+"{}.npz".format(_exclude_food)
	print("Save checkpoint npz as: {}".format(_npz1))
	np.savez(_npz1, weight=weight, bias=bias)

