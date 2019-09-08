#!/usr/bin/env python

"""
Initiates the services `getAction` and `publishLoss`.

TODO:
  * Run this script as a node in a launch script
"""

import argparse
from conban_spanet.algoserver import create_server
from conban_spanet.conbanalg import ContextualBanditAlgo, epsilonGreedy, singleUCB, multiUCB, MultiArmedUCB
import os
import rospy

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
