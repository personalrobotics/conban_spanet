#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import random

NUM_FEATURES = 2048
NUM_ACTIONS = 6

class DatasetDriver:
    def __init__(self, dataset_npz, N=10):
        """
        @param food_type: string specifying excluded food item, e.g. "strawberry"
        @param N: Number of food items to have on the plate at a time
        """
        self.N = N

        # Load Dataset
        dataset = np.load(dataset_npz)
        data = np.vstack((dataset["train"], dataset["test"]))
        num_data = data.shape[0]
        
        # Collect Features / Exp. Reward w/o bias
        self.features = data[:, :NUM_FEATURES]
        self.expected_loss = data[:, NUM_FEATURES:]

        assert self.expected_loss.shape[1] == NUM_ACTIONS

        # Train Pi_Star
        print("Training Pi Star...")

        # Add bias
        self.features_bias = np.pad(self.features, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
        assert self.features_bias.shape[1] == NUM_FEATURES + 1
        assert self.features_bias.shape[0] == self.features.shape[0]

        self.pi_star, _, _, _ = np.linalg.lstsq(self.features_bias, self.expected_loss, rcond=None)
        assert self.pi_star.shape == (NUM_FEATURES + 1, NUM_ACTIONS)
        print("Pi Star Trained!")

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
        rand = np.random.random((self.N, 6))
        ret = np.ones((self.N, 6))

        ret[rand < self.expected_loss[self.plate, :]] = 1

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

        ret = np.argmin(expected_loss, axis=1).reshape((self.N, 1))

        assert ret.shape == (self.N, 1)
        return np.copy(self.pi_star)

    def resample(self, idx):
        """
        @param idx: integer in [0, N-1], food item to re-sample once successfully acquired
        """
        if len(self.unseen_food_idx) <= 0:
            return False
        idx_new = random.sample(self.unseen_food_idx, 1)[0]
        self.unseen_food_idx.remove(idx_new)

        self.plate[idx] = idx_new
        return True



