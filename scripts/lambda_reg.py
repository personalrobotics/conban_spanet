#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import random

from bite_selection_package.config import spanet_config as config
#from get_success_rate import get_pi_and_loss,get_train_test_unseen, get_expected_loss
from conban_spanet.utils import *

NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6

use_dr = False if config.dr_csv==None else True

# Load Dataset
print("Loading dataset...")
data, data_test = get_train_test_unseen()
num_data = data.shape[0]

# Collect Features / Exp. Reward w/o bias
features = data[:, :NUM_FEATURES]

features_test = data_test[:, :NUM_FEATURES]

print("Calculating expected loss...")
expected_loss, expected_loss_test=get_expected_loss(data,data_test,use_dr)

# Add bias
features_bias = pad_feature(features)
assert features_bias.shape[1] == NUM_FEATURES + 1
assert features_bias.shape[0] == num_data
features_bias_test = pad_feature(features_test)

# Train Pi_Star
for lamb in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
	print("Training Pi Star with %f" % lamb)
	_, loss = get_pi_and_loss(features_bias, expected_loss, features_bias_test, expected_loss_test, lamb)
	print("Pi Star Expected Loss: ", loss)
	print("Bootstrapping...")
	bootstrap = np.zeros(100)
	for b in range(100):
		ids = np.random.choice(np.arange(num_data), size=num_data)
		features_bootstrap = features_bias[ids, :]
		expected_bootstrap = expected_loss[ids, :]
		_, loss = get_pi_and_loss(features_bootstrap, expected_bootstrap, features_bias_test, expected_loss_test, lamb)
		bootstrap[b] = loss

	bootstrap = np.sort(bootstrap)
	print("95 upper bound: %f" % bootstrap[95])
	print("95 lower bound: %f" % bootstrap[5])
	print("Mean: %f" % np.mean(bootstrap))
	print()