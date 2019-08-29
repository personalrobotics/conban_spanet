#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np


from bite_selection_package.config import spanet_config as config

NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6
NUM_OUT = 10

DATASET_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET = "dr_dataset_n512_wo_banana_honeydew.npz"
SPANET = "spanet_untrained_dr_n512_wo_banana_honeydew.npz"

use_train = False

def main():
    print("Loading dataset and weights...")
    # Load Dataset
    dataset = np.load(os.path.join(DATASET_DIR, DATASET))
    if use_train:
        data = np.vstack((dataset["train"], dataset["test"]))
    else:
        data = dataset["test"]
    num_data = data.shape[0]

    print("Testing on %d Samples!" % num_data)

    features = data[:, :NUM_FEATURES]
    expected_loss = data[:, NUM_FEATURES:]

    # Load Weights
    loaded = np.load(SPANET)
    weight = loaded["weight"]
    bias = loaded["bias"]

    print("Calculate expected success rate vectors...")

    out = features @ (weight.T) + bias
    assert out.shape == (num_data, NUM_OUT)

    pred = out[:, -NUM_ACTIONS:]
    assert pred.shape == (num_data, NUM_ACTIONS)

    print("Calculated expected success rate on dataset...")
    argmax = np.argmax(pred, axis=1).T

    losses = np.choose(argmax, expected_loss.T)
    assert len(losses) == num_data

    mean_loss = np.mean(losses)
    print("Expected Loss: " + str(mean_loss))
    print("Expected Success Rate: " + str(1.0 - mean_loss))


if __name__ == '__main__':
    main()