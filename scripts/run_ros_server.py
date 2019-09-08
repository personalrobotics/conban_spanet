#!/usr/bin/env python

"""
Initiates the services `getAction` and `publishLoss`.

TODO:
  * Refactor
    - Move heavy-lifting code to `src/food_detector`
    - Move `srv` to `src/food_detector`
  * Run this script as a node in a launch script
  * Use arguments to define `algo`
"""

import rospy
from conban_spanet.conbanalg import singleUCB

from food_detector.srv import GetAction, PublishLoss, GetActionResponse, PublishLossResponse

SERVER_NAME = 'conban_spanet_server'

def _handle_get_action(req, algo):
    p_t = algo.explore(req.features)

    # Sample Action
    _, K = p_t.shape
    p_t_flat = p_t.reshape((-1,))
    sample_idx = np.random.choice(K, p=p_t_flat)
    a_t = sample_idx % K

    return srv.GetActionResponse(a_t, p_t_flat)

def _handle_publish_loss(req, algo):
    try:
        p_t = np.expand_dims(req.p_t, axis=0)
        algo.learn(req.features, 0, req.a_t, req.loss, p_t)
    except:
        return False
    return True

def start_get_action(algo):
    """Starts the `GetAction` service with a given algorithm"""
    def handle_wrapper(req):
        _handle_get_action(req, algo)
    rospy.Service('GetAction', GetAction, handle_wrapper)

def start_publish_loss(algo):
    """Starts the `PublishLoss` service with a given algorithm"""
    def handle_wrapper(req):
        _handle_publish_loss(req, algo)
    rospy.Service('PublishLoss', PublishLoss, handle_wrapper)

def create_server(algo, server_name=SERVER_NAME):
    """
    Creates the algorithm server with a given algorithm.
    Provides the services `GetAction` and `PublishLoss`.
    """
    rospy.init_node(SERVER_NAME)
    start_get_action(algo)
    start_publish_loss(algo)

if __name__ == '__main__':
    # TODO: add options for other CONBANs
    algo = singleUCB(N=1, alpha=0.05, gamma=0)
    create_server(algo)
    try:
        print('Running {} server'.format(SERVER_NAME))
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print('\nShutting down {}...'.format(SERVER_NAME))
