#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import numpy as np
import time

from conban_spanet.environment import Environment
from conban_spanet.conbanalg import *

from bite_selection_package.config import spanet_config as config

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--food_type', default="strawberry",
                    type=str, help="which food item to exclude")

    ap.add_argument('-ho', '--horizon', default=100,
                    type=int, help="how long to run the experiment")
    
    ap.add_argument('-n', '--N', default=15,
                    type=int, help="how many food items in the plate")
    ap.add_argument('-a', '--algo', default="greedy",
    				type=str, help="how many food items in the plate")

    args = ap.parse_args()

    # Validation: make sure args.food_type is in config.items
    try:
        if (args.food_type in config.items) == False:
            raise ValueError("Invalid food_type")
    except ValueError as ve:
        print(ve)


    # Initialize ContextualBanditAlgo
    if  args.algo == "greedy":
        algo = ContextualBanditAlgo(N=args.N)
    elif args.algo == "epsilon":
    	epsilon = input("Set epsilon: ")
    	algo = epsilonGreedy(N=args.N, epsilon=epsilon)
    	args.algo += "_epsilon_" +str(epsilon)
    elif args.algo == "singleUCB":
    	gamma = float(input("Set gamma: "))
    	algo = singleUCB(N=args.N, gamma=gamma)
    	args.algo += "_gamma_"+str(gamma)

    # Initialize Environment
    envir = Environment(args.N)
    
    # Run Environment using args.horizon
    start = time.time()
    cost_algo, cost_spanet = envir.run(algo, args.horizon)
    end = time.time()
    print("Time taken: ", end-start)

    # Store returned lists to CSV for later plotting
    # previous_dir =  os.path.dirname(os.getcwd())
    previous_dir = os.getcwd()
    result_dir = os.path.join(previous_dir, "results")
    cost_to_output = np.array([cost_algo, cost_spanet])
    cost_to_output  = cost_to_output.T
    output_file_name = args.algo+"_N_"+str(args.N)+"_T_"+str(args.horizon)+".csv"
    output_file_name = os.path.join(result_dir, output_file_name)
    print("Saved output file to ", output_file_name)
    np.savetxt(output_file_name, cost_to_output, delimiter=',')

