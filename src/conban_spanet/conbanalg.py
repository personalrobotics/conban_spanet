from .utils import oracle
import numpy as np
import os


LAMB_DEFAULT = 10

from bite_selection_package.config import spanet_config as config
#from get_success_rate import get_train_test_seen, get_expected_loss
from conban_spanet.utils import *

N_FEATURES = 2048 if config.n_features==None else config.n_features


class MultiArmedUCB(object):

    def __init__(self, N, K=6, lambd=LAMB_DEFAULT, d=N_FEATURES, init=" ", pi_0=None,
                T=1000, delta=0.1):
        self.N = N
        self.K = K
        self.T = T
        self.delta = delta
        "Initialization of the policy"
        self.n_t = np.ones(K)
        self.mu_hat_t = np.zeros(K)

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        T = self.T
        delta = self.delta
        p = np.zeros((N, K))
        n = np.random.choice(N) # it really does not matter choose which because they are all strawberry isolated
        argmin_action = np.argmin(self.mu_hat_t - np.sqrt( np.log(2*K*T/delta)/(2*self.n_t) ) )
        p[n, argmin_action] = 1
        return p

    def learn(self, features_t, n_t, a_t, c_t, p_t):
        "Update self.n_t and self.mu_hat_t"
        #"Here we use 0/-1 for success/fail to make the upper bound algo valid"
        #r_t = -c_t
        curr_n_t = self.n_t[a_t]
        curr_mu_hat_t = self.mu_hat_t[a_t]
        # Update
        self.mu_hat_t[a_t] = (curr_mu_hat_t * curr_n_t + c_t) / (curr_n_t + 1)
        self.n_t[a_t] = curr_n_t + 1

class ContextualBanditAlgo(object):
    "N: number of food pieces"

    def __init__(self, N, K=6, lambd=LAMB_DEFAULT, d=N_FEATURES, 
        init="pi_null", pi_0=None):
        self.N = N
        self.K = K

        "Initialization of the policy"
        self.theta = np.zeros((K, d+1))
        self.A = np.array([np.eye(d+1) for i in range(K)]) * lambd
        self.b = np.zeros((K, d+1))
        self.lambd = lambd
        # self.pi = [theta, A, b]

        if init == "pi_null":
            spanet_training_data, spanet_testing_data = get_train_test_seen()
            y_loss, _ = get_expected_loss(spanet_training_data, spanet_testing_data,type="seen")
            feature_train = pad_feature(spanet_training_data[:,:N_FEATURES])
            feature_test = pad_feature(spanet_testing_data[:,:N_FEATURES])

            self.theta = get_pi_and_loss(feature_train,y_loss)
            self.theta = self.theta.T
            assert (self.theta.shape == (K,d+1))
            # Data format: [features, action_idx, loss, food_idx, background_idx]
            # Need the action encoding to be 0-5
            

            for i in range(K):
                data_at_action_i = spanet_training_data[spanet_training_data[:,-4] == i]
                n_data_i = data_at_action_i[:, :-4].shape[0]
                print("#Data for action  {} is {}".format(i, n_data_i)) # DEBUG
                if n_data_i==0:
                    print("No data in action {}".format(i))# DEBUG
                    continue
                X_i = np.ones((n_data_i, d+1))
                X_i[:,1:] = data_at_action_i[:, :-4]
                # y_i = data_at_action_i[:, -3]
                y_i = y_loss[spanet_training_data[:,-4] == i, i]
                # y_i = 
                A_i = np.dot(X_i.T, X_i)
                b_i = np.dot(X_i.T, y_i)
                self.A[i] += A_i
                self.b[i] = b_i
            #     self.theta[i] = np.linalg.solve(A_i + self.lambd * np.eye(d+1), b_i)

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        "Default to be greedy"
        dist = np.dot(features_t, self.theta.T)  # size of N * K
        K = self.K
        N = self.N
        argmin_index = np.argmin(dist)
        argmin_x, argmin_y = argmin_index // K, argmin_index % K
        p = np.zeros((N, K))
        p[argmin_x, argmin_y] = 1
        return p

    def learn(self, features_t, n_t, a_t, c_t, p_t):
        oracle(self, features_t[n_t, :], a_t, c_t, p_t[n_t, a_t])

    def expected_loss(self, driver):
        features = driver.features_bias_test
        expected_loss = driver.expected_loss_test

        pred = np.dot(features,(self.theta.T))
        assert pred.shape == (features.shape[0], self.K)

        argmin = np.argmin(pred, axis=1).T

        losses = np.choose(argmin, expected_loss.T)
        return np.mean(losses)



class epsilonGreedy(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=LAMB_DEFAULT, d=N_FEATURES, init="pi_null", pi_0=None, epsilon=0.1):
        "Default epsilon is 0.1"
        super(epsilonGreedy,self).__init__(N, K, lambd, d, init, pi_0)
        self.epsilon = epsilon

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        epsilon = self.epsilon

        dist = np.dot(features_t, self.theta.T)  # size of N * K
        #argmax_x, argmax_y = argmax_index // K, argmax_index % K

        n = np.random.choice(N)
        argmin_k = np.argmin(dist, axis=1)[n]
        prob_vector_n = np.zeros(K) + epsilon / K
        prob_vector_n[argmin_k] += 1 - epsilon
        p = np.zeros((N, K))
        p[n, :] = prob_vector_n
        return p

    "learn is just a call to oracle, which is same as the superclass"




class singleUCB(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=LAMB_DEFAULT, d=N_FEATURES, init="pi_null", pi_0=None,
                 alpha=0.1, gamma=0.1,dr=True):
        "Default epsilon is 0.1"
        super(singleUCB,self).__init__(N, K, lambd, d, init, pi_0)
        self.alpha = alpha
        self.gamma = gamma
        # self.A = np.array([np.eye(d+1) for i in range(K)]) * lambd
        # self.b = np.zeros((K, d+1))
        
        print("Initializing UCB")
        # Initialize using the SPANet training matrix
        
        
        self.Ainv = np.array([np.eye(d+1) for i in range(K)]) * (1.0/lambd)
        print()

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        alpha = self.alpha
        gamma = self.gamma
        lambd = self.lambd
        while True:
            n = np.random.choice(N)
            phi_n = features_t[n,:]
            lcb = np.dot(self.theta, phi_n) - alpha * np.sqrt( np.dot(np.dot(phi_n.T , self.Ainv) , phi_n))
            assert len(lcb) == K
            if np.amin(lcb) >= gamma:
                continue
            else:
                break
        p = np.zeros((N, K))
        p[n, np.argmin(lcb)] = 1
        return p
        
    def learn(self, features_t, n_t, a_t, c_t, p_t):
        super(singleUCB,self).learn(features_t, n_t, a_t, c_t, p_t)
        self.Ainv[a_t] = np.linalg.inv(self.A[a_t])

    def expected_loss(self, driver):
        features = driver.features_bias
        expected_loss = driver.expected_loss
        num_data = features.shape[0]

        lcb = np.zeros((num_data, self.K))
        for a in range(self.K):
            A = self.Ainv[a, :, :]
            lcb[:, a] = np.dot(features, (self.theta.T[:, a])) - self.alpha * np.sqrt(np.einsum('ij,ji->i', features, (np.dot(A , features.T))))

        argmin = np.argmin(lcb, axis=1).T
        # print(argmin.shape)
        # print(argmin)
        # assert argmin.shape == (num_data, )

        losses = np.choose(argmin, expected_loss.T)
        # print(losses.shape)
        # print(losses)
        # assert losses.shape == (num_data, 1)
        return np.mean(losses)


class multiUCB(singleUCB):
    def __init__(self, N, K=6, lambd=LAMB_DEFAULT, d=N_FEATURES, init="pi_null", pi_0=None,
                 alpha=0.1):
        super(multiUCB,self).__init__(N, K, lambd, d, init, pi_0, alpha)

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        alpha = self.alpha

        lcb = np.zeros((N, K))

        for n in range(N):
            phi_n = features_t[n, :]
            lcb[n, :] = np.dot(self.theta, phi_n) - alpha * np.sqrt( np.dot(np.dot(phi_n.T, self.Ainv) , phi_n))

        p = np.zeros((N, K))
        p[np.unravel_index(np.argmin(lcb), lcb.shape)] = 1
        return p
        
        
        "learn is just a call to oracle, which is same as the superclass"
