#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from conban_spanet.spanet_driver import SPANetDriver
from .utils import test_oracle

class Environment(object):
    def __init__(self, N, d=2048, food_type="strawberry", loc_type="isolated",
        synthetic=True):
        self.N = N
        self.features = np.ones((N, d+1))
        self.driver = SPANetDriver(food_type, loc_type, N, synthetic)

        self.features[:, 1:] = self.driver.get_features()

    def run(self, algo, T):
        N = self.N
        costs_algo = []
        costs_spanet = []
        pi_star_choice_hist = []
        pi_choice_hist = []

        N = algo.N
        K = algo.K
        X_to_test = np.array([[] for i in range(K)])
        y_to_test = np.array([[] for i in range(K)])

        # Run algorithm for T time steps
        for t in range(T):
            if t % 200 == 0:
                print("Now at horzion", t)

            if t == 400:
                test_oracle(algo, )
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

            #print("n_t: " + str(n_t))
            #print("pi_star: " + str(pi_star))

            cost_SPANet = costs[n_t, pi_star]
            cost_algo = costs[n_t, a_t]
            pi_star_choice_hist.append(pi_star)
            pi_choice_hist.append(a_t)

            # Learning
            algo.learn(self.features, n_t, a_t, cost_algo, p_t)
            if t < 400:
                # Update X_to_test, y_to_test
                X_to_test[a_t].append(self.features[n_t, :] / np.sqrt(p_t))
                y_to_test[a_t].append(cost_algo/ np.sqrt(p_t))
            # Replace successfully acquired food item
            if (cost_algo == 0):
                self.driver.resample(n_t)
                self.features[:, 1:] = self.driver.get_features()

            # Record costs for future use
            costs_algo.append(cost_algo)
            costs_spanet.append(cost_SPANet)

        return (costs_algo, costs_spanet,pi_star_choice_hist,pi_choice_hist)
