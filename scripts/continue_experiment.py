#!/usr/bin/env python

"""
Calls service "publishLoss" with all previous data in folder

TODO:
  * Run this script as a node in a launch script
"""

import argparse
from conban_spanet.srv import PublishLoss
import os
import rospy
import rospkg
from bite_selection_package.config import spanet_config as config
from conban_spanet.conbanalg import LAMB_DEFAULT
import numpy as np

N_FEATURES = 2048 if config.n_features==None else config.n_features

rospack = rospkg.RosPack()

if __name__ == '__main__':
    rospy.init_node("continue_experiment")
    ap = argparse.ArgumentParser()

    ap.add_argument('-t', '--trial', default=6,
            type=int, help='which trial num to continue')

    args = ap.parse_args()

    # Init server client
    rospy.wait_for_service('PublishLoss')
    publish_loss = rospy.ServiceProxy('PublishLoss', PublishLoss)
    try:
        # Loop through CSV files
        path = os.path.join(rospack.get_path('conban_spanet'), "online_robot_result/{}_f{}_l{}_trial{}".format(config.excluded_item,N_FEATURES,LAMB_DEFAULT,args.trial))
        print("Cycling Through Path: " + path)
        for filename in os.listdir(path):
            data = np.loadtxt(filename, delimiter=',')
            features = data[:-2]
            action = int(data[-2])
            loss = float(data[-1])
            p_t = [0.0] * 6
            p_t[action] = 1.0
            resp1 = publish_loss(features, action, loss, p_t)
            print("Response: " + str(resp1))
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))
