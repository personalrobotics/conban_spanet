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

def test_oracle(CBAlgo, X_to_test, y_to_test):
	K = CBAlgo.K
	lambd = CBAlgo.lambd
	d=2048
	theta_to_test = np.zeros((K, d+1))
	for i in range(K):
		X_i = np.array(X_to_test[i])
		y_i = np.array(y_to_test[i])
		A = np.dot(X_i.T, X_i) + lambd*np.eye(d+1)
		B = np.dot(X_i.T, y_i)
		theta_to_test[i] = np.linalg.solve(A,B)

	if np.abs(np.sum(CBAlgo.theta - theta_to_test) )> 0.1:
		print("The test sum is {}. Got problem in oracle.".format(np.abs(np.sum(CBAlgo.theta - theta_to_test) )))
	else:
		print("The test sum is {}. Oracle test passed".format(np.abs(np.sum(CBAlgo.theta - theta_to_test) )))