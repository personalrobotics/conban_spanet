#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet
from bite_selection_package.config import spanet_config as config
from bite_selection_package.model.spanet_dataset import SPANetDataset


class SPANetDriver:
    def __init__(self, food_type="strawberry", loc_type='isolated', N=10, synthetic=False):
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
        config.test_list_filepath = os.path.join(config.dataset_dir, 'test.txt')
        #checkpoint_file = "/home/conban/conban_ws/src/bite_selection_package/checkpoint/food_spanet_all_rgb_wall_ckpt_best.pth"
        #checkpoint = torch.load(checkpoint_file)
        checkpoint = torch.load(config.checkpoint_best_filename)
        self.spanet_star.load_state_dict(checkpoint['net'])

        # Load Dataset
        exp_mode = 'test'
        config.excluded_item = food_type
        if food_type is None:
            exp_mode = 'normal'
        config.set_project_prefix()
        self.dataset = SPANetDataset(
        	loc_type = loc_type,
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
        self.features = np.zeros((N, 2048))
        self.success_rates = np.zeros((N, 6))
        self.pi_star = np.zeros((N, 1))

        for i in range(N):
            idx = np.random.randint(0, self.dataset.num_samples)
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
            features (1x2048): vector of features
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
        gv.resize((1, 6))
        pv = pred.cpu().detach()[0][4:].numpy()
        pv.resize((1, 6))
        features = feat_tf.cpu().detach()[0].numpy()
        features.resize((1, 2048))

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
        idx_new = np.random.randint(0, self.dataset.num_samples)
        pv, gv, features = self._sample_dataset(idx_new)
        self.features[idx, :] = features
        self.pi_star[idx, 0] = np.argmax(pv)
        self.success_rates[idx, :] = gv



