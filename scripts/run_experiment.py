#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse

from conban_spanet.environment import Environment
from conban_spanet.conbanalg import *

from bite_selection_package.config import spanet_config as config

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--food_type', default="strawberry",
                    type=str, help="which food item to exclude")

    ap.add_argument('-ho', '--horizon', default=1000,
                    type=int, help="how long to run the experiment")
    args = ap.parse_args()

    # Validation: make sure args.food_type is in config.items

    # Initialize ContextualBanditAlgo

    # Initialize Environment

    # Run Environment using args.horizon

    # Store returned lists to CSV for later plotting
