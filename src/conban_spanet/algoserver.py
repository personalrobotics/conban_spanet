"""Provides methods for running the `GetAction` and `PublishLoss` ROS services"""

from conban_spanet.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse
import numpy as np
import rospy
import time, os
import rospkg

rospack = rospkg.RosPack()

import traceback

from bite_selection_package.config import spanet_config as config
from conban_spanet.conbanalg import LAMB_DEFAULT

N_FEATURES = 2048 if config.n_features==None else config.n_features

SERVER_NAME = 'conban_spanet_server'

#0 = Initial Tests
#1 = First experiment test
#2 = Unknown
#3 = Second experiment test
#4 = Experiment! (Failure)
#5 = Target practice
#6 = Experiment!
#7 = Rerun Carrot
#8 = Greedy
trial_no = 8

def _handle_get_action(req, algo, verbose=True):
    if verbose:
        print('GetAction: called with len(features)={}'.format(len(req.features)))

    # Unflatten features.
    features = np.expand_dims(req.features, axis=0)
    assert features.shape == (algo.N, N_FEATURES+1)

    p_t = algo.explore(features)

    # Sample Action
    _, K = p_t.shape
    p_t_flat = list(p_t.reshape((-1,)))
    sample_idx = np.random.choice(K, p=np.array(p_t_flat))
    a_t = sample_idx % K

    assert p_t_flat[a_t] > 0.99

    if verbose:
        print('GetAction: responding with a_t={} and len(p_t)={}'.format(a_t, len(p_t_flat)))

    return GetActionResponse(a_t, p_t_flat)

def _handle_publish_loss(req, algo, verbose=True):
    if verbose:
        print('PublishLoss: called with len(features)={} a_t={} loss={} len(p_t)={}'.format(len(req.features), req.a_t, req.loss, len(req.p_t)))
    try:
        # Unflatten p_t and features.
        p_t = np.expand_dims(req.p_t, axis=0)
        features = np.expand_dims(req.features, axis=0)
        # Save output result
        output_row = np.hstack((features, np.array([[req.a_t, req.loss]])))
        assert (output_row.shape == (1, N_FEATURES+3)), "Bad shape for output!"
        
        path = os.path.join(rospack.get_path('conban_spanet'), "online_robot_result/{}_f{}_l{}_trial{}".format(config.excluded_item,N_FEATURES,LAMB_DEFAULT,trial_no))

        if not (os.path.isdir(path)): 
            # start_time = time.time()
            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
        file_name = "time_{}.csv".format(time.time())
        print("Saving file: " + str(os.path.join(path,file_name)))
        np.savetxt(os.path.join(path,file_name), output_row, delimiter=",")
        # Learning
        algo.learn(features, 0, req.a_t, req.loss, p_t)
    except:
        print("ERROR:")
        traceback.print_exc()
        return PublishLossResponse(success=False)
    return PublishLossResponse(success=True)

def start_get_action(algo, verbose=True):
    """Starts the `GetAction` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_get_action(req, algo, verbose=verbose)
    rospy.Service('GetAction', GetAction, handle_wrapper)

def start_publish_loss(algo, verbose=True):
    """Starts the `PublishLoss` service with a given algorithm"""
    def handle_wrapper(req):
        return _handle_publish_loss(req, algo, verbose=verbose)
    rospy.Service('PublishLoss', PublishLoss, handle_wrapper)

def create_server(algo, server_name=SERVER_NAME, verbose=True):
    """
    Creates the algorithm server with a given algorithm.
    Provides the services `GetAction` and `PublishLoss`.
    """
    rospy.init_node(SERVER_NAME)
    start_get_action(algo, verbose=verbose)
    start_publish_loss(algo, verbose=verbose)
