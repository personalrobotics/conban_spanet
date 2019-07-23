#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from conban_spanet.dataset_driver import DatasetDriver
from .utils import test_oracle

class Environment(object):
    def __init__(self, dataset_file, N, d=2048):
        self.N = N
        self.features = np.ones((N, d+1))
        self.driver = DatasetDriver(dataset_file, N)

        self.features[:, 1:] = self.driver.get_features()

    def run(self, algo, T, time, time_prev):
        N = self.N
        costs_algo = []
        costs_spanet = []
        pi_star_choice_hist = []
        pi_choice_hist = []

        N = algo.N
        K = algo.K

        # X_to_test = [[] for i in range(K)]
        # y_to_test = [[] for i in range(K)]

        # Run algorithm for T time steps
        for t in range(T):
            if t % 10 == 0:
                time_now = time.time()
                print("Now at horzion", t, " Time taken is ", time_now - time_prev)
                time_prev = time_now
            #if t == 400:
            #    test_oracle(algo, X_to_test, y_to_test)
            # Exploration / Exploitation
            p_t = algo.explore(self.features)

            # Sample Action
            _, K = p_t.shape
            p_t_flat = p_t.reshape((-1,))
            sample_idx = np.random.choice(N*K, p = p_t_flat)
            n_t, a_t = sample_idx // K, sample_idx % K

            # Get Costs
            costs = self.driver.sample_loss_vector()
            pi_star = int(self.driver.get_pi_star()[n_t][0])

            cost_SPANet = costs[n_t, pi_star]
            cost_algo = costs[n_t, a_t]
            pi_star_choice_hist.append(pi_star)
            pi_choice_hist.append(a_t)

            # Learning
            algo.learn(self.features, n_t, a_t, cost_algo, p_t)
            

            # Record costs for future use
            costs_algo.append(cost_algo)
            costs_spanet.append(cost_SPANet)

            # Replace successfully acquired food item
            if (cost_algo == 0):
                if not self.driver.resample(n_t):
                    print("Exhausted all food items!")
                    break
                self.features[:, 1:] = self.driver.get_features()

        return (costs_algo, costs_spanet,pi_star_choice_hist,pi_choice_hist)
