#!/usr/bin/env python

"""
Initiates the services `getAction` and `publishLoss`.

TODO:
  * Refactor
    - Move heavy-lifting code to `src/conban_spanet`
    - Move `srv` to `src/conban_spanet`
  * Run this script as a node in a launch script
  * Use arguments to define `algo`
"""

import argparse
from conban_spanet.conbanalg import ContextualBanditAlgo, epsilonGreedy, singleUCB, multiUCB, MultiArmedUCB
from conban_spanet.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse
import numpy as np
import os
import rospy

SERVER_NAME = 'conban_spanet_server'

def _handle_get_action(req, algo):
    # Unflatten features.
    features = np.expand_dims(req.features, axis=0)

    p_t = algo.explore(features)

    # Sample Action
    _, K = p_t.shape
    p_t_flat = list(p_t.reshape((-1,)))
    sample_idx = np.random.choice(K, p=p_t_flat)
    a_t = sample_idx % K

    return GetActionResponse(a_t, p_t_flat)

def _handle_publish_loss(req, algo):
    try:
        # Unflatten p_t and features.
        p_t = np.expand_dims(req.p_t, axis=0)
        features = np.expand_dims(req.features, axis=0)

        # Learning
        algo.learn(features, 0, req.a_t, req.loss, p_t)
    except:
        return PublishLossResponse(success=False)
    return PublishLossResponse(success=True)

def start_get_action(algo):
    """Starts the `GetAction` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_get_action(req, algo)
    rospy.Service('GetAction', GetAction, handle_wrapper)

def start_publish_loss(algo):
    """Starts the `PublishLoss` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_publish_loss(req, algo)
    rospy.Service('PublishLoss', PublishLoss, handle_wrapper)

def create_server(algo, server_name=SERVER_NAME):
    """
    Creates the algorithm server with a given algorithm.
    Provides the services `GetAction` and `PublishLoss`.
    """
    rospy.init_node(SERVER_NAME)
    start_get_action(algo)
    start_publish_loss(algo)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('-a', '--algo', default="greedy",
		    type=str, help='algorithm to use: greedy, epsilon, singleUCB, multiUCB, UCB')
    ap.add_argument('-alp', '--alpha', default=0.05,
                    type=float, help='alpha for LinUCB')
    # TODO: Is this a good default for delta?
    ap.add_argument('-d', '--delta',default=0,
                    type=float, help="delta for MultiArmedUCB")
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

    elif args.algo == 'epsilon':
        print('Epsilon-greedy contextual bandit chosen with epsilon {}'.format(args.epsilon))
	algo = epsilonGreedy(N=N, epsilon=args.epsilon)

    elif args.algo == 'singleUCB':
        print('Single UCB chosen with alpha {} and gamma {}'.format(args.alpha, args.gamma))
	algo = singleUCB(N=N, alpha=args.alpha, gamma=args.gamma)

    elif args.algo == 'multiUCB':
        print('Multi UCB chosen')
	algo = multiUCB(N=N)

    elif args.algo == 'UCB':
        print('Multiarmed UCB chosen with delta {}'.format(args.delta))
        # TODO: should we set the horizon, T?
	algo = MultiArmedUCB(N=N, delta=args.delta)

    else:
        exit('"{}" is not a valid algorithm type'.format(args.algo))

    create_server(algo)

    try:
        print('Running {} server'.format(SERVER_NAME))
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print('\nShutting down {}...'.format(SERVER_NAME))
