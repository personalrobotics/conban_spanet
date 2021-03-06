#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import random

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet
from bite_selection_package.config import spanet_config as config
from bite_selection_package.model.spanet_dataset import SPANetDataset

N_FEATURES = 2048 if config.n_features==None else config.n_features
N_ACTIONS = 6


class SPANetDriver:
    def __init__(self, food_type="strawberry", loc_type='isolated', N=10, 
        synthetic=False, use_wo_spanet=False, seen=True):
        """
        @param food_type: string specifying excluded food item, e.g. "strawberry"
        @param N: Number of food items to have on the plate at a time
        """
        self.N = N
        self.synthetic = synthetic

        # Load SPANet
        self.spanet_star = SPANet(use_rgb=config.use_rgb, use_depth=config.use_depth, use_wall=config.use_wall)
        if config.use_cuda:
            self.spanet_star = self.spanet_star.cuda()
        self.spanet_star.eval()
        if use_wo_spanet:
            config.excluded_item = food_type
        else:
            config.excluded_item = None
        config.set_project_prefix()
        # XM note: config.project_dir was false.
        config.project_dir = "/home/conban/conban_ws/src/bite_selection_package"
        config.dataset_dir = os.path.join(config.project_dir, 'data/skewering_positions_{}'.format(config.project_keyword))
        config.img_dir = os.path.join(config.dataset_dir, 'cropped_images')
        config.depth_dir = os.path.join(config.dataset_dir, 'cropped_depth')
        config.ann_dir = os.path.join(config.dataset_dir, 'annotations')
        config.success_rate_map_path = os.path.join(
                            config.dataset_dir,
                            'identity_to_success_rate_map_{}.json'.format(config.project_keyword))

        config.pretrained_dir = os.path.join(config.project_dir, 'pretrained')
        config.checkpoint_filename = os.path.join(
                config.project_dir, 'checkpoint/{}_ckpt.pth'.format(config.project_prefix))
        config.checkpoint_best_filename = os.path.join(
                config.project_dir, 'checkpoint/{}_ckpt_best.pth'.format(config.project_prefix))

        print("Loading Checkpoint: " + config.checkpoint_best_filename)
        config.test_list_filepath = os.path.join(config.dataset_dir, 'test.txt')
        #checkpoint_file = "/home/conban/conban_ws/src/bite_selection_package/checkpoint/food_spanet_all_rgb_wall_ckpt_best.pth"
        #checkpoint = torch.load(checkpoint_file)
        checkpoint = torch.load(config.checkpoint_best_filename)
        self.spanet_star.load_state_dict(checkpoint['net'])

        # Load Dataset
        exp_mode = 'test'
        #config.excluded_item = "banana_honeydew_grape_spinach_cauliflower_strawberry_broccoli_kiwi"
        config.excluded_item = None  #food_type
        if config.excluded_item is None:
            exp_mode = 'normal'
        config.set_project_prefix()

        # only include isolated
        assert config.test_list_filepath, 'invalid list_filepath'
        with open(config.test_list_filepath, 'r') as f_list:
            ann_filenames = list(map(str.strip, f_list.readlines()))
        if loc_type=="isolated":
            ann_filenames_to_include =[]
            for ann_filename in ann_filenames:
                if ann_filename.find('isolated') >= 0:
                    ann_filenames_to_include.append(ann_filename)
        else:
            ann_filenames_to_include =ann_filenames
        self.dataset = SPANetDataset(
        	ann_filenames = ann_filenames_to_include,
            img_dir=config.img_dir,
            depth_dir=config.depth_dir,
            ann_dir=config.ann_dir,
            success_rate_map_path=config.success_rate_map_path,
            img_res=config.img_res,
            list_filepath=config.test_list_filepath,
            train=False,
            exp_mode=exp_mode,
            excluded_item=config.excluded_item,
            transform=transforms.Compose([transforms.ToTensor()]),
            use_rgb=config.use_rgb,
            use_depth=config.use_depth,
            use_wall=config.use_wall)

        # Sample N food items
        self.features = np.zeros((N, N_FEATURES))
        self.success_rates = np.zeros((N, N_ACTIONS))
        self.pi_star = np.zeros((N, 1))

        # Create sampe set
        self.unseen_food_idx = set()
        self.unseen_food_idx.update([i for i in range(self.dataset.num_samples)])

        for i in range(N):
            idx = random.sample(self.unseen_food_idx, 1)[0]
            self.unseen_food_idx.remove(idx)
            pv, gv, features = self._sample_dataset(idx)
            self.features[i, :] = features
            self.pi_star[i, 0] = np.argmax(pv)
            if self.synthetic:
                self.success_rates[i, :] = pv
            else:
                self.success_rates[i, :] = gv


    def _sample_dataset(self, idx):
        """
        @return:
            pv (1x6): SPANet's expected success rate for each action
            gv (1x6): Ground-truth success rate for each action
            features (1xN_FEATURES): vector of features
        """

        # Pre-processing
        rgb, depth, gt_vector, loc_type = self.dataset[idx]
        rgb = torch.stack([rgb]) if rgb is not None else None
        depth = torch.stack([depth]) if depth is not None else None
        gt_vector = torch.stack([gt_vector])
        loc_type = torch.stack([loc_type])
        if config.use_cuda:
            rgb = rgb.cuda() if rgb is not None else None
            depth = depth.cuda() if depth is not None else None
            loc_type = loc_type.cuda()

        # Run SPANet
        pred, feat_tf = self.spanet_star(rgb, depth, loc_type)

        # Post-process to Numpy
        gv = gt_vector.cpu().detach()[0][4:].numpy()
        gv.resize((1, N_ACTIONS))
        pv = pred.cpu().detach()[0][4:].numpy()
        pv.resize((1, N_ACTIONS))
        features = feat_tf.cpu().detach()[0].numpy()
        features.resize((1, N_FEATURES))

        return pv, gv, features


    def sample_loss_vector(self):
        """
        @return (Nx6): loss vector sampled from ground-truth success rates
        """
        rand = np.random.random((self.N, 6))
        ret = np.ones((self.N, 6))

        ret[rand < self.success_rates] = 0
        return ret

    def get_features(self):
        """
        @return (Nx2048): features for all food items
        """
        return np.copy(self.features)

    def get_pi_star(self):
        """
        @return (Nx1): SPAnet's action recommendation for each food item
        """
        return np.copy(self.pi_star)

    def resample(self, idx):
        """
        @param idx: integer in [0, N-1], food item to re-sample once successfully acquired
        """
        if len(self.unseen_food_idx) <= 0:
            return False
        idx_new = random.sample(self.unseen_food_idx, 1)[0]
        self.unseen_food_idx.remove(idx_new)
        pv, gv, features = self._sample_dataset(idx_new)
        self.features[idx, :] = features
        self.pi_star[idx, 0] = np.argmax(pv)
        self.success_rates[idx, :] = gv
        return True



