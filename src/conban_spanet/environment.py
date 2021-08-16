#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from conban_spanet.dataset_driver import DatasetDriver
from .utils import test_oracle

from bite_selection_package.config import spanet_config as config

N_FEATURES = 2048 if config.n_features==None else config.n_features

class Environment(object):
    def __init__(self, N, d=N_FEATURES):
        self.N = N
        self.features = np.ones((N, d+1))
        print("Initializing Unseen Dataset")
        self.driver = DatasetDriver(N)

        self.features[:, 1:] = self.driver.get_features()

    def run(self, algo, T, time, time_prev):
        N = self.N
        costs_algo = []
        costs_spanet = []
        costs_pi_null = []
        pi_star_choice_hist = []
        pi_choice_hist = []

        N = algo.N
        K = algo.K

        num_failures = 0
        MAX_FAILURES = 1

        expected_srs = []
        loss_list = []

        # X_to_test = [[] for i in range(K)]
        # y_to_test = [[] for i in range(K)]

        # Run algorithm for T time steps
        for t in range(T):
            # Feb 22: We do not care about mean expected loss anymore
            # exp_loss = algo.expected_loss(self.driver)
            # loss_list.append(exp_loss)
            # expected_srs.append(1.0 - exp_loss)

            # Logging
            if t % 10 == 0:
                time_now = time.time()
                print("Now at horzion", t, " Time taken is ", time_now - time_prev)
                time_prev = time_now
                

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
            #pi_star = np.argmin(costs[n_t, :])
            #pi_null = int(self.driver.get_pi_null(algo)[n_t][0])
            pi_null = np.random.choice(len(costs[n_t, :]))

            cost_SPANet = costs[n_t, pi_star]
            cost_algo = costs[n_t, a_t]
            cost_pi_null = costs[n_t, pi_null]
            pi_star_choice_hist.append(pi_star)
            pi_choice_hist.append(a_t)

            # Learning
            algo.learn(self.features, n_t, a_t, cost_algo, p_t)
            #for a in range(6):
            #    algo.learn(self.features, n_t, a, costs[n_t, a], np.ones(p_t.shape))


            # Record costs for future use
            costs_algo.append(cost_algo)
            costs_spanet.append(cost_SPANet)
            costs_pi_null.append(cost_pi_null)

            if t % 10 == 0:
                # 
                # # Getting expectd loss of algorithm
                # print("Expected loss is : " + str(exp_loss))
                print("cumulative loss is :"+str(np.sum(costs_algo)))
                time_now = time.time()
                print("Time Taken: ", time_now - time_prev)
                time_prev = time_now

            # Replace successfully acquired food item
            # Or give up after some amount of time.
            """
            if (cost_algo == 1):
                num_failures += 1
                if num_failures >= MAX_FAILURES:
                    cost_algo = 0

            if (cost_algo == 0):
                num_failures = 0
                if not self.driver.resample(n_t):
                    print("Exhausted all food items!")
                    break
                self.features[:, 1:] = self.driver.get_features()
            """
            if not self.driver.resample(n_t):
                print("Exhausted all food items!")
                break
            self.features[:, 1:] = self.driver.get_features()

        
        # print("Calculating the last loss of algo...")
        # print("Algo cost: ", np.sum(costs_algo))
        # print("pi star cost: ", np.sum(costs_spanet))
        # Getting expected loss of algorithm
        # 22 Feb 2020: we do not need expected loss anymore
        # exp_loss = algo.expected_loss(self.driver)
        # print("Expected Loss: " + str(exp_loss))
        # expected_srs.append(1.0 - exp_loss)
        time_now = time.time()
        print("Time Taken: ", time_now - time_prev)
        time_prev = time_now

        print("Cumulative pi null loss is:        ", np.sum(costs_pi_null))
        
        pi_star_loss =self.driver.pi_star_loss

        return (costs_algo, costs_spanet,pi_star_choice_hist,pi_choice_hist,expected_srs,loss_list, pi_star_loss)
