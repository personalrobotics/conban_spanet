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



class DatasetDriver:
    def __init__(self, N=10):
        """
        @param food_type: string specifying excluded food item, e.g. "strawberry"
        @param N: Number of food items to have on the plate at a time
        """
        self.N = N

        # Load Dataset
        # dataset = np.load(dataset_npz)
        # data = dataset["train"]
        # data_test = dataset["test"]
        data, data_test = get_train_test_unseen()
        #data_seen_test = np.load("scripts/dr_dataset_dr_seen.npz")["test"]
        num_data = data.shape[0]
        
        # Collect Features / Exp. Reward w/o bias
        self.features = data[:, :NUM_FEATURES]
        #self.expected_loss = data[:, NUM_FEATURES:]

        self.features_test = data_test[:, :NUM_FEATURES]
        print()
        print("Number of testing item:: ", self.features_test.shape[0])
        print()
        #self.expected_loss_test = data_test[:, NUM_FEATURES:]
        self.expected_loss,self.expected_loss_test=get_expected_loss(data,data_test,use_dr)

        #self.features_seen_test = data_seen_test[:, :NUM_FEATURES]
        #self.expected_loss_seen_test = data_seen_test[:, NUM_FEATURES:]

        assert self.expected_loss.shape[1] == NUM_ACTIONS

        # Train Pi_Star
        print("Training Pi Star with lamb=50...")

        # Add bias

        self.features_bias = pad_feature(self.features)
        assert self.features_bias.shape[1] == NUM_FEATURES + 1
        assert self.features_bias.shape[0] == num_data
        self.features_bias_test = pad_feature(self.features_test)
        #self.features_bias_test = np.pad(self.features_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
        #self.features_bias_seen_test = np.pad(self.features_seen_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))

        self.pi_star, loss = get_pi_and_loss(self.features_bias, self.expected_loss, self.features_bias_test, self.expected_loss_test, 10)
        print("Pi Star Expected Loss: ", loss)

        # Train on 1/6 of the data samples
        """
        print("Testing on 1/4 of the dataset")
        test_indices = np.random.randint(num_data, size=int(num_data / 4))
        for lamb in (0.01, 50, 100, 500, 1000, 5000):
            print("Testing on ", lamb)
            _, loss = get_pi_and_loss(self.features_bias[test_indices, :], self.expected_loss[test_indices, :], self.features_bias_test, self.expected_loss_test, lamb)
            print("Expected Loss: ", loss)
        """

        # Add N items to plate
        self.plate = []

        # Create sample set
        self.unseen_food_idx = set()
        self.unseen_food_idx.update([i for i in range(num_data)])

        for i in range(N):
            idx = random.sample(self.unseen_food_idx, 1)[0]
            self.unseen_food_idx.remove(idx)
            self.plate.append(idx)
        print("Number of unseen food:", len(self.unseen_food_idx))


    def sample_loss_vector(self):
        """
        @return (Nx6): loss vector sampled from ground-truth success rates
        """
        #rand = np.random.random((self.N, 6))
        #ret = np.zeros((self.N, 6))
        #ret[rand < self.expected_loss[self.plate, :]] = 1

        ret = np.copy(self.expected_loss[self.plate, :])

        assert ret.shape == (self.N, NUM_ACTIONS)
        return ret

    def get_features(self):
        """
        @return (Nx2048): features for all food items on plate
        """
        ret = np.copy(self.features[self.plate, :])

        assert ret.shape == (self.N, NUM_FEATURES)
        return ret

    def get_pi_star(self):
        """
        @return (Nx1): SPAnet's action recommendation for each food item
        """
        expected_loss = np.dot(self.features_bias[self.plate, :], self.pi_star)
        assert expected_loss.shape == (self.N, NUM_ACTIONS)

        #print("Expected Loss: " + str(expected_loss))

        ret = np.argmin(expected_loss, axis=1).reshape((self.N, 1))

        assert ret.shape == (self.N, 1)
        return ret

    def resample(self, idx):
        """
        @param idx: integer in [0, N-1], food item to re-sample once successfully acquired
        """

        #print("Re-sampling food!")

        if len(self.unseen_food_idx) <= 0:
            return False
        idx_new = random.sample(self.unseen_food_idx, 1)[0]
        self.unseen_food_idx.remove(idx_new)

        self.plate[idx] = idx_new
        return True



