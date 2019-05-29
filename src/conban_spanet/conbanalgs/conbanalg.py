import numpy as np

def array_map(f, x):
    return np.array(list(map(f, x)))

def oracle(CBAlgo, feature_n_t, a_t, c_t, p_a_t):
    "CBAlgo: ContextualBanditAlgo object"
    A_a_t, b_a_t = CBAlgo.A[a_t], CBAlgo.b[a_t]
    A_a_t += np.outer(feature_n_t, feature_n_t) / p_a_t
    b_a_t += c_t / p_a_t * feature_n_t
    theta_a_t = np.linalg.solve(A_a_t, b_a_t)
    CBAlgo.A[a_t] = A_a_t
    CBAlgo.b[a_t] = b_a_t
    CBAlgo.theta[a_t] = theta_a_t


from utils import oracle
class ContextualBanditAlgo():
    "N: number of food pieces"
    "Environment contains a lot of information"
    def __init__(self, E, N, K = 6, lambd = 0.1, d=2048, init=" ", pi_0=None):
        self.N = N
        self.K = K
        "Plate is contained in E"
        self.E = E
        
        "Initialization of the policy"
        self.theta = np.zeros((K, d+1))
        self.A = np.array([np.eye(d+1) for i in range(K)]) * lambd
        self.b = np.zeros((K, d+1))
        # self.pi = [theta, A, b]
        if init == "etc":
            self.theta, self.A, self.b = pi_0
            
    def explore(self, features_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        "Default to be greedy"
        dist = np.dot(features_t, self.theta.T) # size of N * K
        K = self.K
        N = self.N
        argmax_index = np.argmax(dist)
        argmax_x, argmax_y = argmax_index // K, argmax_index % K
        p = np.zeros((N, K))
        p[argmax_x, argmax_y] = 1
        return p
        
    
    def learn(features_t, n_t, a_t, c_t, p_t):
        oracle(self, features_t[n_t], a_t, c_t, p_t[n_t, a_t])

        
class epsilonGreedy(ContextualBanditAlgo):
    def __init__(self, E, N, K = 6, lambd = 0.1, d=2048, init=" ", pi_0=None, epsilon=0.1):
        "Default epsilon is 0.1"
        super().__init__(E, N, K, lambd, d, init, pi_0)
        self.epsilon = epsilon
    
    def explore(self, feature_t):
        "p: N * K dimensional prob. matrix, with the sum to 1"
        "features_t: (N * 2049) feature matrix"
        K = self.K
        N = self.N
        epsilon = self.epsilon
        
        dist = np.dot(features_t, self.theta.T) # size of N * K
        argmax_x, argmax_y = argmax_index // K, argmax_index % K
        
        n = np.random.choice(N)
        argmax_k = np.argmax(dist, axis = 1)[n]
        prob_vector_n = np.zeros(K) + epsilon / K
        prob_vector_n[argmax_k] += 1 - epsilon
        p = np.zeros((N, K))
        p[n,  :] = prob_vector_n
        return p
    
    "learn is just a call to oracle, which is same as the superclass"
