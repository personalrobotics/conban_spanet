#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import torch


from bite_selection_package.config import spanet_config as config
from conban_spanet.utils import *
from conban_spanet.conbanalg import LAMB_DEFAULT


NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6
NUM_OUT = 10


def main():
    print("Loading dataset and weights...")
    # Load Dataset
    # dataset = np.load(os.path.join(DATASET_DIR, DATASET))
    # if use_train:
    #     data = np.vstack((dataset["train"], dataset["test"]))
    # else:
    #     data = dataset["test"]
    # num_test_data = data.shape[0]

    # Separate the unseen food item
    data_train, data_test = get_train_test_unseen(isolated=False)
    num_test_data = data_test.shape[0]

    print("Testing on %d Samples!" % num_test_data)


    features_test = data_test[:, :NUM_FEATURES]
    _, expected_loss = get_expected_loss(data_train,data_test,dr=use_dr)


    
    if feat_version == "spanet":
        # Load Weights
        _exclude_food = '_wo_{}'.format(config.excluded_item) if config.excluded_item else ''
    else:
        _exclude_food = '_wo_banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi'
    
    _feat_str = "_n"+str(NUM_FEATURES) if config.n_features!=None else ''
    _dr_str = "_dr" if use_dr else ''
    _npz1 = "spanet_untrained{}{}".format(_dr_str,_feat_str)+"{}.npz".format(_exclude_food)
    if os.path.exists(_npz1)==False:
        print("Creating npz for SPANet checkpoint")
        ckpt_path = "/home/conban/conban_ws/src/bite_selection_package/checkpoint/food_spanet_all"+_feat_str+"_rgb_wall"+_exclude_food+"_dr_ckpt_best.pth"
        checkpoint = torch.load(ckpt_path) 
        print("Get checkpoint from: {}".format(ckpt_path))
        weight = checkpoint['net']['final.weight'].cpu().numpy() 
        bias = checkpoint['net']['final.bias'].cpu().numpy()
        print("Save checkpoint npz as: {}".format(_npz1))
        np.savez(_npz1, weight=weight, bias=bias)
    else:
        print("Loading npz for SPANet checkpoint: {}".format(_npz1))
    loaded = np.load(_npz1)
    weight = loaded["weight"]
    bias = loaded["bias"]
    
    print("Calculate expected success rate vectors...")
    out = np.dot(features_test, (weight.T) )+ bias
    assert out.shape == (num_test_data, NUM_OUT)

    pred = out[:, -NUM_ACTIONS:]
    assert pred.shape == (num_test_data, NUM_ACTIONS)

    print("Calculated expected success rate on dataset...")
    argmax = np.argmax(pred, axis=1).T

    losses = np.choose(argmax, expected_loss.T)
    assert len(losses) == num_test_data
     

    mean_loss = np.mean(losses)
    print("Expected Loss: " + str(mean_loss))
    print("Expected Success Rate: " + str(1.0 - mean_loss))

    
    # Collect Features / Exp. Reward w/o bias
    features = data_train[:, :NUM_FEATURES]
    #self.expected_loss = data[:, NUM_FEATURES:]

    num_data = features.shape[0]
    print("Number of testing item:: ", features_test.shape[0])
    print()
    #self.expected_loss_test = data_test[:, NUM_FEATURES:]
    expected_loss,expected_loss_test=get_expected_loss(data_train,data_test,dr=True)

    #self.features_seen_test = data_seen_test[:, :NUM_FEATURES]
    #self.expected_loss_seen_test = data_seen_test[:, NUM_FEATURES:]

    assert expected_loss.shape[1] == NUM_ACTIONS

    # Train Pi_Star
    print("Training Pi Star with lamb={}...".format(LAMB_DEFAULT))

    # Add bias

    features_bias = pad_feature(features)
    assert features_bias.shape[1] == NUM_FEATURES + 1
    assert features_bias.shape[0] == num_data
    features_bias_test = pad_feature(features_test)
    #self.features_bias_test = np.pad(self.features_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))
    #self.features_bias_seen_test = np.pad(self.features_seen_test, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))

    pi_star, loss = get_pi_and_loss(features_bias, expected_loss, features_bias_test, expected_loss_test, LAMB_DEFAULT)
    print("Pi Star Expected Loss: ", loss)

if __name__ == '__main__':
    main()
