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
