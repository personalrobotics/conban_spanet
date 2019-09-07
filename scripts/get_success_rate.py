#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import torch


from bite_selection_package.config import spanet_config as config
from conban_spanet.utils import *



NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6
NUM_OUT = 10

# FOOD_NAME_TO_INDEX = {
#     "_banana_": 0,
#     "_honeydew_": 1,
#     "_grape_": 2,
#     "_spinach_": 3,
#     "_cauliflower_": 4,
#     "_strawberry_": 5,
#     "_broccoli_": 6,
#     "_kiwi_": 7,
#     "_cherry_tomato_": 8,
#     "_kale_": 9,
#     "_lettuce_": 10,
#     "_celery_": 11,
#     "_bell_pepper_": 12,
#     "_carrot_": 13,
#     "_cantaloupe_": 14,
#     "_apple_": 15
# }

# BACKGROUND_TO_INDEX = {
#     "isolated": 0,
#     "wall": 1,
#     "lettuce+": 2
# }

# def get_food_name():
#     return FOOD_NAME_TO_INDEX


# def get_background_name():
#     return BACKGROUND_TO_INDEX



#DATASET_DIR = os.path.dirname(os.path.realpath(__file__))
#DATASET = "dr_dataset_n512_wo_banana_honeydew.npz"
#SPANET = "spanet_untrained_dr_n512_wo_banana.npz"

# EXCLUDED_FOOD = config.excluded_item
# FOOD_NAME_TO_INDEX = get_food_name()
# food_items = EXCLUDED_FOOD.split("_")
# food_items = list(map(lambda x:"_"+x+"_",food_items))
# food_item_indices = list(map(lambda x:FOOD_NAME_TO_INDEX[x],food_items))
# food_item_indices_seen = list(set(FOOD_NAME_TO_INDEX.values()) - set(food_item_indices))

# # use_wall = config.use_wall
# use_train = False
# use_dr = False if config.dr_csv==None else True

# Either "spanet" or "all". "spanet"
# "spanet" means feature from the spanet based on the current unseen
# "all" means feature from spanet based on 8 seen
# feat_version = "spanet" 
# data_path = "/home/conban/conban_ws/src/conban_spanet/barnes_dataset/curr_spanet"

# if feat_version == "all":
#     data_path = "/home/conban/conban_ws/src/conban_spanet/barnes_dataset"
    

# def calculate_expected_loss(dataset, failure_rate_dict,dr):
#     n = dataset.shape[0]
#     expected_loss = np.empty((n, NUM_ACTIONS))
#     for i in range(n):
#         food_item_selected = int(dataset[i,-2])
#         action_selected = int(dataset[i,-4])
#         loss_i = dataset[i,-3] # 1 is failure, 0 is success
#         failure_rate_i = failure_rate_dict[food_item_selected]
#         if dr:
#             failure_rate_i[action_selected] += 6*(loss_i - failure_rate_i[action_selected])
#         failure_rate_i[failure_rate_i>=1] = 0.99
#         failure_rate_i[failure_rate_i<=0] = 0.01
#         expected_loss[i] = failure_rate_i
#     return expected_loss

# def get_expected_loss(data_train, dataset, dr=True,type="unseen"):
#     'Input: data_train: numpy object of barnes training dataset, for success rate calculation'
#     '       dataset: test data set. We need to provide expeced loss for that'
#     if type=="unseen":
#         indices = food_item_indices
#     elif type=="seen":
#         indices = food_item_indices_seen
#     else:
#         raise(Exception("Unspecified type"))
#     failure_rate_dict = dict()
#     for food_item_index in indices:
#         failure_rate_food = np.empty(NUM_ACTIONS)
#         for action in range(NUM_ACTIONS):
#             failure_rate=float(np.mean(data_train[:,-3][data_train[:,-4]==action]))
#             failure_rate_food[action] = failure_rate
#         #print("Failure_rate_raw: ",failure_rate_food)
#         if food_item_index == 0: # For banana, we copy over 0 to 90 degree
#             for action in [1,3,5]:
#                 failure_rate_food[action] = failure_rate_food[action-1]
#         failure_rate_dict[food_item_index] = failure_rate_food

#     return (calculate_expected_loss(data_train,failure_rate_dict,dr),
#             calculate_expected_loss(dataset,failure_rate_dict,dr) )

# def pad_feature(features):
#     return np.pad(features, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))

# def get_pi_and_loss(features_bias, expected_loss, features_bias_test=None, expected_loss_test=None, lamb=0):
#     pi= np.linalg.solve(features_bias.T.dot(features_bias) + lamb * np.identity(features_bias.shape[1]), features_bias.T.dot(expected_loss))
#     assert pi.shape == (NUM_FEATURES + 1, NUM_ACTIONS)
    
#     if not (features_bias_test is None):
#         pred = np.dot(features_bias_test, pi)
#         assert pred.shape == (features_bias_test.shape[0], NUM_ACTIONS)

#         argmin = np.argmin(pred, axis=1).T

#         losses = np.choose(argmin, expected_loss_test.T)

#         return pi, np.mean(losses)
#     else:
#         return pi

# def retrieve_data_from_food(data, food_item_indices):
#     all_data_list = list(map(lambda x: data[data[:,-2]==x,:],food_item_indices))
#     all_data = all_data_list[0]
#     for dat in all_data_list[1:]:
#         all_data = np.concatenate((all_data, dat),axis=0)
#     return all_data

# def get_train_test_seen():
#     train_file = os.path.join(data_path, "barnes_partial_dataset_train_all.csv")
#     test_file = os.path.join(data_path,"barnes_partial_dataset_test_all.csv")
#     data_train = np.genfromtxt(train_file,delimiter=',')
#     data_test = np.genfromtxt(test_file,delimiter=',')
#     # if use_wall:
#     #     data_train = data_train[data_train[:,-1]==1]
#     #     data = data[data[:,-1]==1]
#     data_train = retrieve_data_from_food(data_train, food_item_indices_seen)
#     data_test = retrieve_data_from_food(data_test, food_item_indices_seen)
#     print("Retrieved {} train seen food".format(data_train.shape[0]))
#     return (data_train,data_test)

# def get_train_test_unseen():
#     train_file = os.path.join(data_path, "barnes_partial_dataset_train_all.csv")
#     test_file = os.path.join(data_path,"barnes_partial_dataset_test_all.csv")
#     data_train = np.genfromtxt(train_file,delimiter=',')
#     data = np.genfromtxt(test_file,delimiter=',')
#     # if use_wall:
#     #     data_train = data_train[data_train[:,-1]==1]
#     #     data = data[data[:,-1]==1]
#     print("Testing these food items: {}".format(EXCLUDED_FOOD.split("_")))
#     data_train = retrieve_data_from_food(data_train, food_item_indices)
#     data = retrieve_data_from_food(data, food_item_indices)

#     return (data_train,data)

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
    data_train, data = get_train_test_unseen()
    num_test_data = data.shape[0]

    print("Testing on %d Samples!" % num_test_data)


    features = data[:, :NUM_FEATURES]
    _, expected_loss = get_expected_loss(data_train,data,dr=use_dr)


    
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
    out = np.dot(features, (weight.T) )+ bias
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


if __name__ == '__main__':
    main()
