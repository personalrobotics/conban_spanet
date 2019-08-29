#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import random

from bite_selection_package.config import spanet_config as config

NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6

def get_pi_star_and_loss(features_bias, expected_loss, features_bias_test, expected_loss_test, lamb=0):
    pi_star= np.linalg.solve(features_bias.T.dot(features_bias) + lamb * np.identity(features_bias.shape[1]), features_bias.T.dot(expected_loss))
    assert pi_star.shape == (NUM_FEATURES + 1, NUM_ACTIONS)

    pred = features_bias_test @ pi_star
    assert pred.shape == (features_bias_test.shape[0], NUM_ACTIONS)

    argmin = np.argmin(pred, axis=1).T

    losses = np.choose(argmin, expected_loss_test.T)

    return pi_star, np.mean(losses)

class DatasetDriver:
    def __init__(self, dataset_npz, N=10):
        """
        @param food_type: string specifying excluded food item, e.g. "strawberry"
        @param N: Number of food items to have on the plate at a time
        """
        self.N = N

        # Load Dataset
        dataset = np.load(dataset_npz)
        data = dataset["train"]
        data_test = dataset["test"]
        #data_seen_test = np.load("scripts/dr_dataset_dr_seen.npz")["test"]
        num_data = data.shape[0]
        
        # Collect Features / Exp. Reward w/o bias
        self.features = data[:, :NUM_FEATURES]
        self.expected_loss = data[:, NUM_FEATURES:]

        self.features_test = data_test[:, :NUM_FEATURES]
        self.expected_loss_test = data_test[:, NUM_FEATURES:]

        #self.features_seen_test = data_seen_test[:, :NUM_FEATURES]
        #self.expected_loss_seen_test = data_seen_test[:, NUM_FEATURES:]

        assert self.expected_loss.shape[1] == NUM_ACTIONS

        # Train Pi_Star
        print("Training Pi Star with lamb=50...")

        # Add bias
        self.features_bias = np.pad(self.features, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
        assert self.features_bias.shape[1] == NUM_FEATURES + 1
        assert self.features_bias.shape[0] == num_data

        self.features_bias_test = np.pad(self.features_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
        #self.features_bias_seen_test = np.pad(self.features_seen_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))

        self.pi_star, loss = get_pi_star_and_loss(self.features_bias, self.expected_loss, self.features_bias_test, self.expected_loss_test, 10)
        print("Pi Star Expected Loss: ", loss)

        # Train on 1/6 of the data samples
        """
        print("Testing on 1/4 of the dataset")
        test_indices = np.random.randint(num_data, size=int(num_data / 4))
        for lamb in (0.01, 50, 100, 500, 1000, 5000):
            print("Testing on ", lamb)
            _, loss = get_pi_star_and_loss(self.features_bias[test_indices, :], self.expected_loss[test_indices, :], self.features_bias_test, self.expected_loss_test, lamb)
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
        expected_loss = self.features_bias[self.plate, :] @ self.pi_star
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



