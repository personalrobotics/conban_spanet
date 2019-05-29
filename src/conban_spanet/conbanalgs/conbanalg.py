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
        dist = np.dot(features_t, self.theta.T) "N * K"
        K = self.K
        N = self.N
        argmax_index = np.argmax(dist)
        argmax_x, argmax_y = argmax_index // K, argmax_index % K
        p = np.zeros((N, K))
        p[argmax_x, argmax_y] = 1
        return p
        
    
    def learn(features_t, n_t, a_t, c_t, p_t):
        oracle(self, features_t[n_t], a_t, c_t, p_t[n_t, a_t])
