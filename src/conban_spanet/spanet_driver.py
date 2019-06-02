#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.config import spanet_config as config
from bite_selection_package.model.spanet_dataset import SPANetDataset


class SPANetDriver:
    def __init__(self, food_type="strawberry", N=10):
        """
        @param food_type: string specifying excluded food item, e.g. "strawberry"
        @param N: Number of food items to have on the plate at a time
        """

        # Load SPANet
        self.spanet_star = SPANet(use_rgb=config.use_rgb, use_depth=config.use_depth, use_wall=config.use_wall)
        config.excluded_item = None
        config.set_project_prefix()
        checkpoint = torch.load(config.checkpoint_best_filename)
        self.spanet_star.load_state_dict(checkpoint['net'])

        # Load Dataset
        config.excluded_item = food_type
        config.set_project_prefix()
        self.dataset = SPANetDataset(
            img_dir=config.img_dir,
            depth_dir=config.depth_dir,
            ann_dir=config.ann_dir,
            success_rate_map_path=config.success_rate_map_path,
            img_res=config.img_res,
            list_filepath=config.test_list_filepath,
            train=False,
            exp_mode='test',
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
            gt_vector = gt_vector.cuda()
            loc_type = loc_type.cuda()
        
        # Run SPANet
        pred, feat_tf = spanet(rgb, depth, loc_type)

        # Post-process to Numpy
        gv = gt_vector.cpu().detach()[0][4:].numpy().resize((1, 6))
        pv = pred.cpu().detach()[0][4:].numpy().resize((1, 6))
        features = feat_tf.cpu().detach()[0].numpy().resize((1, 2048))

        return pv, gv, features


    def sample_loss_vector(self):
        """
        @return (Nx6): loss vector sampled from ground-truth success rates
        """
        rand = np.random.random((N, 6))
        ret = np.ones((N, 6))

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



