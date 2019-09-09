#!/usr/bin/env python

"""Tests the `GetAction` and `PublishLoss` services"""

import argparse
from bite_selection_package.config import spanet_config as config
from conban_spanet.algoserver import create_server
from conban_spanet.conbanalg import ContextualBanditAlgo, epsilonGreedy, singleUCB, multiUCB, MultiArmedUCB
from conban_spanet.dataset_driver import DatasetDriver
from conban_spanet.srv import GetAction, PublishLoss
import numpy as np
import os
import rospy
import signal
import time

SAVE_FILE = 'server_features.csv'
N_FEATURES = 2048 if config.n_features==None else config.n_features

last_features = None

def signal_handler(sig, frame):
    global last_features
    print()
    print('Caught SIGINT')
    if last_features is not None:
        print('Saving features to {}'.format(SAVE_FILE))
        np.savetxt(SAVE_FILE, last_features, delimiter=',')
    print('Exiting...')
    exit(0)

def run(algo, T, time, time_prev, N, K):
    global last_features

    # Create service proxies for `GetAction` and `PublishLoss`.
    print('Creating service proxies for GetAction and PublishLoss...')
    rospy.wait_for_service('GetAction')
    try:
        get_action = rospy.ServiceProxy('GetAction', GetAction)
    except rospy.ServiceException as e:
        exit('Failed to create proxy for GetAction: {}'.format(e))
    rospy.wait_for_service('PublishLoss')
    try:
        publish_loss = rospy.ServiceProxy('PublishLoss', PublishLoss)
    except rospy.ServiceException as e:
        exit('Failed to create proxy for PublishLoss: {}'.format(e))

    # This code is derived from conbanalg.environment.Environment.
    driver = DatasetDriver(N)
    features = np.ones((N, N_FEATURES+1))
    features[:, 1:] = driver.get_features()

    costs_algo = []
    costs_spanet = []
    pi_star_choice_hist = []
    pi_choice_hist = []

    num_failures = 0
    MAX_FAILURES = 1

    expected_srs = []

    # Run algorithm for T time steps.
    for t in range(T):
        if t % 10 == 0:
            time_now = time.time()
            print('Now at horzion', t, ' Time taken is ', time_now - time_prev)
            time_prev = time_now

            # Getting expected loss of algorithm.
            print('Calculating expected loss of algo...')
            exp_loss = algo.expected_loss(driver)
            print('Expected Loss: ' + str(exp_loss))
            expected_srs.append(1.0 - exp_loss)
            time_now = time.time()
            print('Time Taken: ', time_now - time_prev)
            time_prev = time_now

        # Sample Action
        feat_flat = list(features.reshape((-1,)))
        try:
            resp = get_action(feat_flat)
        except rospy.ServiceException as e:
            exit('Failed to get response from GetAction: {}'.format(e))
        n_t = 0
        a_t = resp.a_t
        p_t = np.expand_dims(resp.p_t, axis=0)

        # Get Costs
        costs = driver.sample_loss_vector()
        pi_star = int(driver.get_pi_star()[n_t][0])

        cost_SPANet = costs[n_t, pi_star]
        cost_algo = costs[n_t, a_t]
        pi_star_choice_hist.append(pi_star)
        pi_choice_hist.append(a_t)

        # Flatten p_t
        p_t_flat = list(p_t.reshape((-1,)))
        try:
            # Learning
            publish_loss(feat_flat, a_t, cost_algo, p_t_flat)
        except rospy.ServiceException as e:
            exit('Failed to get response from PublishLoss: {}'.format(e))

        # Record costs for future use
        costs_algo.append(cost_algo)
        costs_spanet.append(cost_SPANet)

        # Replace successfully acquired food item
        # Or give up after some amount of time.
        if not driver.resample(n_t):
            print('Exhausted all food items!')
            break
        features[:, 1:] = driver.get_features()

        # These features are saved when a SIGINT is caught.
        last_features = features

    # Getting expected loss of algorithm
    print('Calculating expected loss of algo...')
    exp_loss = algo.expected_loss(driver)
    print('Expected Loss: ' + str(exp_loss))
    expected_srs.append(1.0 - exp_loss)
    time_now = time.time()
    print('Time Taken: ', time_now - time_prev)
    time_prev = time_now

    np.savez('expected_srs.npz', srs=np.array(expected_srs))

    return (costs_algo, costs_spanet, pi_star_choice_hist, pi_choice_hist)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('-ho', '--horizon', default=5000,
                    type=int, help='how long to run the experiment')
    ap.add_argument('-a', '--algo', default='greedy',
    		    type=str, help='algorithm to use: greedy, epsilon, singleUCB, multiUCB, UCB')
    ap.add_argument('-alp', '--alpha', default=0.05,
                    type=float, help='alpha for LinUCB')
    # TODO: Is this a good default for delta?
    ap.add_argument('-d', '--delta',default=0,
                    type=float, help='delta for MultiArmedUCB')
    ap.add_argument('-e', '--epsilon', default=0,
                    type=float, help='epsilon for epsilon greedy')
    ap.add_argument('-ga', '--gamma',default=1000, 
                    type=float, help='gamma for singleUCB')
    ap.add_argument('-g', '--gpu', default='0', type=str, help='GPU ID')

    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # We are only looking at one food item. This may change later.
    N = 1

    # Initialize algorithm
    if  args.algo == 'greedy':
        print('Standard contextual bandit chosen')
        algo = ContextualBanditAlgo(N=N)
        prefix = args.algo
    elif args.algo == 'epsilon':
        print('Epsilon-greedy contextual bandit chosen with epsilon {}'.format(args.epsilon))
        algo = epsilonGreedy(N=N, epsilon=args.epsilon)
        prefix = '{}_epsilon_{}'.format(args.algo, args.epsilon)
    elif args.algo == 'singleUCB':
        print('Single UCB chosen with alpha {} and gamma {}'.format(args.alpha, args.gamma))
        algo = singleUCB(N=N, alpha=args.alpha, gamma=args.gamma)
        prefix = '{}_alpha_{}_gamma_{}'.format(args.algo, args.alpha, args.gamma)
    elif args.algo == 'multiUCB':
        print('Multi UCB chosen')
        algo = multiUCB(N=N)
        prefix = args.algo
    elif args.algo == 'UCB':
        print('Multiarmed UCB chosen with delta {}'.format(args.delta))
        algo = MultiArmedUCB(N=N, T=args.horizon, delta=args.delta)
        prefix = '{}_delta_{}'.format(args.algo, args.delta)
    else:
        exit('"{}" is not a valid algorithm type'.format(args.algo))

    signal.signal(signal.SIGINT, signal_handler)

    create_server(algo, verbose=False)

    # Run environment
    start = time.time()
    cost_algo, cost_spanet, pi_star_choice_hist, pi_choice_hist = run(
        algo, args.horizon, time=time, time_prev=start, N=N, K=algo.K)
    end = time.time()
    print('Time taken: {}'.format(end - start))

    # Store returned lists to CSV for later plotting.
    # Now output to regret and choice history.
    previous_dir = os.getcwd()
    result_dir = os.path.join(previous_dir, 'results')
    data_to_output = np.array([cost_algo, cost_spanet, pi_star_choice_hist, pi_choice_hist])
    data_to_output  = data_to_output.T
    output_path = '{}_N_{}_T_{}_wserver.csv'.format(prefix, N, args.horizon)
    output_path = os.path.join(result_dir, output_path)
    print('Saved output file to {}'.format(output_path))
    np.savetxt(output_path, data_to_output, delimiter=',')
