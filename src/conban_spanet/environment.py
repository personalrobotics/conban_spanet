import torch
import torchvision.transforms as transforms

from PIL import Image

import torch
import torch.optim as optim
import torchvision.transforms as transforms

from bite_selection_package.model.spanet import SPANet, DenseSPANet
from bite_selection_package.model.spanet_dataset import SPANetDataset
from bite_selection_package.model.spanet_loss import SPANetLoss
from bite_selection_package.config import spanet_config as config

def Environment(object):
    def __init__(self, N, T, d=2048):
        self.N = N
        self.T = T
        self.features = np.ones((N, d+1))
        self.true_actions = np.zeros((N))
        self.SPANetDriver = SPANetDriver()
        
    def sampleSingleFood(self):
        "sample a image, and a ground truth action"
        pass
    
    def sampleSingleFoodAtPosition(self, i):
        "sample pair = self.sampleSingleFood()(image, ground truth action)"
        self.features[i, 1:] = "map a image to a feature vector"
        self.true_actions[i] = "ground truth action"

    def run(self, algo):
        "Sample a whole plate"
        N = self.N
        T = self.T
        for i in range(N):
            self.sampleSingleFoodAtPosition(i)
            
        for t in range(T):
            p_t = algo.explore(self.features)
            N, K = p_t.shape
            p_t = p_t.reshape((-1,))
            sample_idx = np.random.choice(N*K, p = p_t)
            n_t, a_t = sample_idx // K, sample_idx % K
            cost_SPAnet = "TODO: SPAnet only use 2048d hence [i,1:] self.features[i, 1:]"
            cost_algo = (a_t == self.true_actions[n_t])
            algo.learn(features_t, n_t, a_t, cost_algo, p_t)
            
            if (cost_algo == 0):
                "Succefully caucht this food. Resample a new food item"
                self.sampleSingleFoodAtPosition(i)
                
