from utils import oracle
import numpy as np

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
        oracle(self, features_t[n_t], a_t, c_t, p_t[n_t, a_t])


class epsilonGreedy(ContextualBanditAlgo):
    def __init__(self, N, K=6, lambd=0.1, d=2048, init=" ", pi_0=None, epsilon=0.1):
        "Default epsilon is 0.1"
        super().__init__(N, K, lambd, d, init, pi_0)
        self.epsilon = epsilon

    def explore(self, feature_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        epsilon = self.epsilon

        dist = np.dot(features_t, self.theta.T)  # size of N * K
        argmax_x, argmax_y = argmax_index // K, argmax_index % K

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
                 gamma=0.1, delta=0.1, R=3, S=2):
        "Default epsilon is 0.1"
        super().__init__(N, K, lambd, d, init, pi_0)
        self.gamma = gamma
        self.delta = delta
        self.R = R
        self.S = S

    def explore(self, feature_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        gamma = self.gamma
        delta = self.delta
        R = self.R
        S = self.S
        lambd = self.lambd
        while True:
            n = np.random.choice(N)
            phi_n = feature_t[n,:]
            lcb = np.zeros(K)
            for a in range(K):
                A_a = self.A[a]
                theta_a = self.theta[a]
                d = A_a.shape[0]
                "Linear programming is used for UCB"
                norm_bound = R * np.sqrt(2 * np.log(np.sqrt(np.linalg.det(A) / np.linalg.det(lambd*np.ones(d))) / delta)) \
                    + np.sqrt(lambd) * S
                x = cp.Variable(d)
                prob = cp.Problem(cp.Minimize(phi_n.T*x),[cp.quad_form(x - theta_a, A) <= norm_bound**2])
                prob.solve()
                
                lcb[a] = prob.value
            if np.min(lcb) >= gamma:
                continue
            else:
                break
        p = np.zeros((N, K))
        p[n, np.argmin(lcb)] = 1
        return p
        
        "learn is just a call to oracle, which is same as the superclass"
