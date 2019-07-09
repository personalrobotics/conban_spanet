from .utils import oracle
import numpy as np


class MultiArmedUCB(object):

    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None,
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
        argmax_action = np.argmax(self.mu_hat_t + np.sqrt( np.log(2*K*T/delta)/(2*self.n_t) ) )
        p[n, argmax_action] = 1
        return p

    def learn(self, features_t, n_t, a_t, c_t, p_t):
        "Update self.n_t and self.mu_hat_t"
        "Here we use 0/-1 for success/fail to make the upper bound algo valid"
        r_t = -c_t
        curr_n_t = self.n_t[a_t]
        curr_mu_hat_t = self.mu_hat_t[a_t]
        # Update
        self.mu_hat_t[a_t] = (curr_mu_hat_t * curr_n_t + r_t) / (curr_n_t + 1)
        self.n_t[a_t] = curr_n_t + 1

class ContextualBanditAlgo(object):
    "N: number of food pieces"

    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None):
        self.N = N
        self.K = K

        "Initialization of the policy"
        self.theta = np.zeros((K, d+1))
        self.A = np.array([np.eye(d+1) for i in range(K)]) * lambd
        self.b = np.zeros((K, d+1))
        self.lambd = lambd
        # self.pi = [theta, A, b]
        if init == "etc":
            self.theta, self.A, self.b = pi_0

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        "Default to be greedy"
        dist = np.dot(features_t, self.theta.T)  # size of N * K
        K = self.K
        N = self.N
        argmax_index = np.argmax(dist)
        argmax_x, argmax_y = argmax_index // K, argmax_index % K
        p = np.zeros((N, K))
        p[argmax_x, argmax_y] = 1
        return p

    def learn(self, features_t, n_t, a_t, c_t, p_t):
        oracle(self, features_t[n_t, :], a_t, c_t, p_t[n_t, a_t])



class epsilonGreedy(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None, epsilon=0.1):
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
        argmax_k = np.argmax(dist, axis=1)[n]
        prob_vector_n = np.zeros(K) + epsilon / K
        prob_vector_n[argmax_k] += 1 - epsilon
        p = np.zeros((N, K))
        p[n, :] = prob_vector_n
        return p

    "learn is just a call to oracle, which is same as the superclass"




class singleUCB(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None,
                 alpha=0.1, gamma=0.1):
        "Default epsilon is 0.1"
        super(singleUCB,self).__init__(N, K, lambd, d, init, pi_0)
        self.alpha = alpha
        self.gamma = gamma

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
            ucb = np.zeros(K)
            for a in range(K):
                A_a = self.A[a]
                theta_a = self.theta[a]
                d = A_a.shape[0]

                # "Linear programming is used for UCB"
                # norm_bound = R * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A_a) / np.linalg.det(lambd*np.eye(d))) / delta)) \
                #     + np.sqrt(lambd) * S
                # x = cp.Variable(d)
                # prob = cp.Problem(cp.Minimize(phi_n.T*x),[cp.quad_form(x - theta_a, A) <= norm_bound**2])
                # prob.solve()
                # ucb[a] = prob.value
                ucb[a] = np.dot(theta_a, phi_n) + alpha * np.sqrt(np.dot(phi_n, np.dot(np.linalg.inv(A_a),phi_n)))
            if np.amax(ucb) <= gamma:
                continue
            else:
                break
        p = np.zeros((N, K))
        p[n, np.argmax(ucb)] = 1
        return p
        
        "learn is just a call to oracle, which is same as the superclass"





class multiUCB(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None,
                 alpha=0.1):
        super(multiUCB,self).__init__(N, K, lambd, d, init, pi_0)
        self.alpha = alpha

    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        alpha = self.alpha

        # Pre-calculate inverses
        A_inv = [None] * K
        for a in range(K):
            A_inv[a] = np.linalg.inv(self.A[a])
            
        # lambd = self.lambd
        ucb = np.zeros((N,K))
        for n in range(N):
            phi_n = features_t[n,:]
            for a in range(K):
                A_a = self.A[a]
                theta_a = self.theta[a]
                d = A_a.shape[0]
                ucb[n, a] = np.dot(theta_a, phi_n) + alpha * np.sqrt(np.dot(phi_n, np.dot(A_inv[a],phi_n)))
        p = np.zeros((N, K))
        argmax_index = np.argmax(ucb)
        argmax_x, argmax_y = argmax_index // K, argmax_index % K
        p[argmax_x, argmax_y] = 1
        return p
        
        "learn is just a call to oracle, which is same as the superclass"
