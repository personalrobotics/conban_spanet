#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import numpy as np
import time
import sys
import signal

from conban_spanet.environment import Environment
from conban_spanet.conbanalg import *

from bite_selection_package.config import spanet_config as config

envir = None

def signal_handler(sig, frame):
    print()
    print("Caught SIGINT")
    if envir is not None:
        print("Saving features to features.csv")
        np.savetxt("features.csv", envir.features, delimiter=",")

    print("Exiting...")
    sys.exit(0)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('-ho', '--horizon', default=5000,
                    type=int, help="how long to run the experiment")
    
    ap.add_argument('-n', '--N', default=15,
                    type=int, help="how many food items in the plate")
    ap.add_argument('-a', '--algo', default="greedy",
    				type=str, help="how many food items in the plate")
    ap.add_argument('-alp', '--alpha', default=0.05,
                    type=float, help="alpha for LinUCB")
    ap.add_argument('-ga', '--gamma',default=1000, 
                    type=float, help="gamma for singleUCB")
    ap.add_argument('-g', '--gpu', default='0', type=str, help="GPU ID")

    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # Initialize ContextualBanditAlgo
    if  args.algo == "greedy":
        algo = ContextualBanditAlgo(N=args.N)
    elif args.algo == "epsilon":
    	epsilon = float(input("Set epsilon: "))
    	algo = epsilonGreedy(N=args.N, epsilon=epsilon)
    	args.algo += "_epsilon_" +str(epsilon)
    elif args.algo == "singleUCB":
    	algo = singleUCB(N=args.N, alpha=args.alpha, gamma=args.gamma)
    	args.algo += "_alpha_"+str(args.alpha)+"_gamma_"+str(args.gamma)
    elif args.algo == "multiUCB":
    	algo = multiUCB(N=args.N)
    elif args.algo == "UCB":
    	delta = float(input("Set delta: "))
    	algo = MultiArmedUCB(N=args.N, T=args.horizon, delta=delta)
    	args.algo += "_delta_"+str(delta)

    # Initialize Environment
    envir = Environment(args.N)

    signal.signal(signal.SIGINT, signal_handler)
    
    # Run Environment using args.horizon
    start = time.time()
    cost_algo, cost_spanet,pi_star_choice_hist,pi_choice_hist = envir.run(algo, args.horizon, 
                        time=time, time_prev=start)
    end = time.time()
    print("Time taken: ", end-start)

    # Store returned lists to CSV for later plotting
    # Now output to regret and choice history
    previous_dir = os.getcwd()
    result_dir = os.path.join(previous_dir, "results")
    data_to_output = np.array([cost_algo, cost_spanet, pi_star_choice_hist,pi_choice_hist])
    data_to_output  = data_to_output.T
    output_file_name = args.algo+"_N_"+str(args.N)+"_T_"+str(args.horizon)+".csv"
    output_file_name = os.path.join(result_dir, output_file_name)
    print("Saved output file to ", output_file_name)
    np.savetxt(output_file_name, data_to_output, delimiter=',')

