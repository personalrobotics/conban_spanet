import os
import numpy as np
from bite_selection_package.config import spanet_config as config
import rospkg

def array_map(f, x):
    return np.array(list(map(f, x)))

FOOD_NAME_TO_INDEX = {
    "_banana_": 0,
    "_honeydew_": 1,
    "_grape_": 2,
    "_spinach_": 3,
    "_cauliflower_": 4,
    "_strawberry_": 5,
    "_broccoli_": 6,
    "_kiwi_": 7,
    "_cherry_tomato_": 8,
    "_kale_": 9,
    "_lettuce_": 10,
    "_celery_": 11,
    "_bell_pepper_": 12,
    "_carrot_": 13,
    "_cantaloupe_": 14,
    "_apple_": 15
}

BACKGROUND_TO_INDEX = {
    "isolated": 0,
    "wall": 1,
    "lettuce+": 2
}

def get_food_name():
    return FOOD_NAME_TO_INDEX


def get_background_name():
    return BACKGROUND_TO_INDEX


NUM_FEATURES = 2048 if config.n_features==None else config.n_features
NUM_ACTIONS = 6
# NUM_OUT = 10


EXCLUDED_FOOD = config.excluded_item
FOOD_NAME_TO_INDEX = get_food_name()
food_items = EXCLUDED_FOOD.split("_")
food_items = list(map(lambda x:"_"+x+"_",food_items))
food_item_indices = list(map(lambda x:FOOD_NAME_TO_INDEX[x],food_items))
food_item_indices_seen = list(set(FOOD_NAME_TO_INDEX.values()) - set(food_item_indices))

# use_wall = config.use_wall
use_train = False
use_dr = False if config.dr_csv==None else True

# Either "spanet" or "all". "spanet"
# "spanet" means feature from the spanet based on the current unseen
# "all" means feature from spanet based on 8 seen
feat_version = "spanet" 
rospack = rospkg.RosPack()
data_path = os.path.join(rospack.get_path('conban_spanet'), "barnes_dataset/curr_spanet/")

if feat_version == "all":
    data_path = os.path.join(rospack.get_path('conban_spanet'), "barnes_dataset/")
    

def calculate_expected_loss(dataset, failure_rate_dict,dr):
    n = dataset.shape[0]
    expected_loss = np.empty((n, NUM_ACTIONS))
    for i in range(n):
        food_item_selected = int(dataset[i,-2])
        action_selected = int(dataset[i,-4])
        loss_i = dataset[i,-3] # 1 is failure, 0 is success

        failure_rate_i = failure_rate_dict[food_item_selected].copy()
        # print(failure_rate_i)
        if dr:
            failure_rate_i[action_selected] += 6*(loss_i - failure_rate_i[action_selected])
        # XM 2019 Nov 22: we want real loss, i.e. take off manual thresholding
        # failure_rate_i[failure_rate_i>=1] = 0.99
        # failure_rate_i[failure_rate_i<=0] = 0.01
        expected_loss[i] = failure_rate_i.copy()
        if (max(abs(expected_loss[i])) > 6.0):
            print(expected_loss[i])
            print("food item selected: ",food_item_selected)
            print("action_selected: ", action_selected)
            print("loss_i: ", loss_i)
            print("failure dict is: ",failure_rate_dict)
        assert max(abs(expected_loss[i])) <= 6.0
    return expected_loss

def get_expected_loss(data_train, dataset, dr=True,type="unseen"):
    'Input: data_train: numpy object of barnes training dataset, for success rate calculation'
    '       dataset: test data set. We need to provide expeced loss for that'
    if type=="unseen":
        indices = food_item_indices
    elif type=="seen":
        indices = food_item_indices_seen
    else:
        raise(Exception("Unspecified type"))
    failure_rate_dict = dict()
    for food_item_index in indices:
        failure_rate_food = np.empty(NUM_ACTIONS)
        for action in range(NUM_ACTIONS):
            dtrain_loss_filter_by_action = data_train[:,-4:][data_train[:,-4]==action]
            failure_rate=float(np.mean(dtrain_loss_filter_by_action[:,-3][dtrain_loss_filter_by_action[:,-2]==food_item_index]))
            # XM 2019 Nov 22
            # failure_rate=float(np.mean(data_train[:,-3][data_train[:,-4]==action]))
            failure_rate_food[action] = failure_rate
        #print("Failure_rate_raw: ",failure_rate_food)
        if food_item_index == 0: # For banana, we copy over 0 to 90 degree
            for action in [1,3,5]:
                failure_rate_food[action] = failure_rate_food[action-1]
        failure_rate_dict[food_item_index] = failure_rate_food
    print("The loss dictionary is ", failure_rate_dict)
    return (calculate_expected_loss(data_train,failure_rate_dict,dr),
            calculate_expected_loss(dataset,failure_rate_dict,dr) )

def pad_feature(features):
    return np.pad(features, ((0, 0), (1, 0)), 'constant', constant_values=(1, 1))

def get_pi_and_loss(features_bias, expected_loss, features_bias_test=None, expected_loss_test=None, lamb=0):
    pi= np.linalg.solve(features_bias.T.dot(features_bias) + lamb * np.identity(features_bias.shape[1]), features_bias.T.dot(expected_loss))
    assert pi.shape == (NUM_FEATURES + 1, NUM_ACTIONS)
    
    if not (features_bias_test is None):
        pred = np.dot(features_bias_test, pi)
        assert pred.shape == (features_bias_test.shape[0], NUM_ACTIONS)

        argmin = np.argmin(pred, axis=1).T

        losses = np.choose(argmin, expected_loss_test.T)

        return pi, np.mean(losses)
    else:
        return pi

def retrieve_data_from_food(data, food_item_indices):
    all_data_list = list(map(lambda x: data[data[:,-2]==x,:],food_item_indices))
    all_data = all_data_list[0]
    for dat in all_data_list[1:]:
        all_data = np.concatenate((all_data, dat),axis=0)
    return all_data

def get_train_test_seen(isolated=False):
    train_file = os.path.join(data_path, "barnes_partial_dataset_train_all.csv")
    test_file = os.path.join(data_path,"barnes_partial_dataset_test_all.csv")
    data_train = np.genfromtxt(train_file,delimiter=',')
    data_test = np.genfromtxt(test_file,delimiter=',')
    if isolated:
        data_train = data_train[data_train[:,-1]==0]
        data_test = data_test[data_test[:,-1]==0]
    data_train = retrieve_data_from_food(data_train, food_item_indices_seen)
    data_test = retrieve_data_from_food(data_test, food_item_indices_seen)
    print("Retrieved {} train seen food".format(data_train.shape[0]))
    return (data_train,data_test)

def get_train_test_unseen(isolated=False):
    train_file = os.path.join(data_path, "barnes_partial_dataset_train_all.csv")
    test_file = os.path.join(data_path,"barnes_partial_dataset_test_all.csv")
    data_train = np.genfromtxt(train_file,delimiter=',')
    data_test = np.genfromtxt(test_file,delimiter=',')
    if isolated:
        data_train = data_train[data_train[:,-1]==0]
        data_test = data_test[data_test[:,-1]==0]
    print("Testing these food items: {}".format(EXCLUDED_FOOD.split("_")))
    data_train = retrieve_data_from_food(data_train, food_item_indices)
    data_test = retrieve_data_from_food(data_test, food_item_indices)
    print("Retrieved {} train unseen food".format(data_train.shape[0]))
    return (data_train,data_test)




def oracle(CBAlgo, feature_n_t, a_t, c_t, p_a_t):
    "CBAlgo: ContextualBanditAlgo object"
    A_a_t, b_a_t = CBAlgo.A[a_t], CBAlgo.b[a_t]
    A_a_t += np.outer(feature_n_t, feature_n_t) / p_a_t
    b_a_t += c_t / p_a_t * feature_n_t
    theta_a_t = np.linalg.solve(A_a_t, b_a_t)
    CBAlgo.A[a_t] = A_a_t
    CBAlgo.b[a_t] = b_a_t
    CBAlgo.theta[a_t] = theta_a_t

def test_oracle(CBAlgo, X_to_test, y_to_test):
	K = CBAlgo.K
	lambd = CBAlgo.lambd
	d=2048
	theta_to_test = np.zeros((K, d+1))
	for i in range(K):
		X_i = np.array(X_to_test[i])
		y_i = np.array(y_to_test[i])
		if len(X_i) >0:
			A = np.dot(X_i.T, X_i) + lambd*np.eye(d+1)
			B = np.dot(X_i.T, y_i)
			theta_to_test[i] = np.linalg.solve(A,B)
		else:
			theta_to_test[i] = 0

	if np.abs(np.sum(CBAlgo.theta - theta_to_test) )> 0.1:
		print("The test sum is {}. Got problem in oracle.".format(np.abs(np.sum(CBAlgo.theta - theta_to_test) )))
	else:
		print("The test sum is {}. Oracle test passed".format(np.abs(np.sum(CBAlgo.theta - theta_to_test) )))

