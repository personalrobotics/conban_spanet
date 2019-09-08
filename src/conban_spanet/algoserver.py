"""Provides methods for running the `GetAction` and `PublishLoss` ROS services"""

from conban_spanet.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse
import numpy as np
import rospy

from bite_selection_package.config import spanet_config as config
N_FEATURES = 2048 if config.n_features==None else config.n_features

SERVER_NAME = 'conban_spanet_server'

def _handle_get_action(req, algo):
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
