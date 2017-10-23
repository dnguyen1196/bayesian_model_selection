"""
This module performs the iterative method based on evidence
function to maximize the evidence function f(t | a, b)
where w ~ N(0, I/a)
We then find MSE using the MAP(mN) formula for prediction

Note that the values that we want to find through the iterations
are alpha and beta

li * beta are the eigenvalues of 1/b A'A

gamma = sum(lambda * b/ (a + lambda * b))

alpha = gamma / mN'mN

Where mN = beta SN A't
SN = (alpha I + b A'A)^-1

1/b = 1/(N - gamma) * sum{ t - mN'A }^2
"""
import numpy as np
import math


"""
    Load data matrix from file
"""
def load_data_matrix(data_file):
    f = open(data_file, "r")
    A = []
    for line in f:
        data = np.array([float(x) for x in line.strip().split(",")])
        A.append(data)
    f.close()
    return A


"""
    Load result vector from file
"""
def load_target_vector(target_file):
    f = open(target_file, "r")
    t = []
    for line in f:
        t.append(float(line.strip()))
    f.close()
    return np.asarray(t)


"""
We need S_N to compute m_N
Where S_N = (alpha * I + beta * Phi'Phi)^-1
We can implicitly find S_N by first finding the eigenvalues of S_N which is diag(alpha + beta * eig(Phi'Phi))
The eigenvectors of S_N are the same as the eigenvectors of Phi'Phi, call them V

Then S_N^-1 = V (1/Sigma) V' where Sigma is the diag(eig) of alpha + beta * eig(Phi'Phi)
"""
def compute_s_n(alpha, beta, eigenvalues, eigenvectors):
    sn_eigen_vals = np.diag(1/(alpha + beta * eigenvalues))
    return np.dot(eigenvectors, np.dot(sn_eigen_vals, np.transpose(eigenvectors)))


"""
Compute m_N
m_N = beta * S_N Phi't
"""
def compute_m_n(alpha, beta, eigenvalues, eigenvectors, phi_transpose_t):
    s_n = compute_s_n(alpha, beta, eigenvalues, eigenvectors)
    return beta * np.dot(s_n, phi_transpose_t)


"""
Note that lambda_i here is the eigen values of Beta Phi'Phi
So it is lambda_i = beta * eig_i
gamma = sum { lambda_i / alpha + lambda_i }
"""
def compute_gamma(alpha, beta, eigenvalues):
    vector = np.divide(beta*eigenvalues, alpha + beta*eigenvalues)
    return np.sum(vector)


"""
1/beta = 1/(N-gamma) sum { t - mn'Phi(x)}
"""
def compute_beta(N, gamma, m_n, Phi, t):
    r = np.subtract(t, np.dot(Phi, m_n))
    beta_inverse = 1/(N-gamma) * np.sum(np.power(r, 2))
    return 1/beta_inverse


"""
    Iterative algorithm to find alpha and beta
"""
def optimize_evidence(Phi, t):
    # Optimization stopping constants
    max_iteration = 100
    eps = 1e-16

    # Initial random alpha and beta
    [alpha,beta] = np.random.uniform(0, 10, [2,1])
    N = float(len(t))

    # Precompte repeated inner product and quantities
    phi_transpose_phi = np.dot(np.transpose(Phi), Phi)
    phi_transpose_t = np.dot(np.transpose(Phi), t)
    eigen_values, eigen_vectors = np.linalg.eig(phi_transpose_phi)

    # Initialize
    alpha_cur, beta_cur = alpha, beta

    for i in range(max_iteration):
        # Compute m_n and gamma
        m_n = compute_m_n(alpha_cur, beta_cur, eigen_values, eigen_vectors, phi_transpose_t)
        gamma = compute_gamma(alpha_cur, beta_cur, eigen_values)

        # Compute updated alpha and beta
        alpha = gamma/np.dot(m_n, m_n)
        beta = compute_beta(N, gamma, m_n, Phi, t)

        # Check stopping criterion
        if abs(alpha - alpha_cur) < eps and abs(beta - beta_cur) < eps:
            break
        alpha_cur = alpha
        beta_cur = beta

    return alpha, beta, m_n, i


"""
Compute the test set mse by first optimizing the evidence function and then 
uses MAP distribution to predict
"""
def compute_optimal_evidence_mse(train_data_file, train_target_file, test_data_file, test_target_file):
    phi = load_data_matrix(train_data_file)
    t = load_target_vector(train_target_file)

    B = load_data_matrix(test_data_file)
    T = load_target_vector(test_target_file)
    alpha, beta, m_n, i = optimize_evidence(phi, t)
    r = np.dot(B, m_n) - T
    return 1/float(len(T)) * np.sum(np.power(r, 2))


"""
Expand the feature space from x (a vector)
"""
def expand_feature_space(x, n):
    augmented = np.ones(shape=[len(x),1],dtype=float)
    for i in range(1,n+1):
        feature = np.power(x, i)
        augmented = np.hstack((augmented, feature))
    return augmented


"""
Optimize and compute MSE in expanded feature space
"""
def compute_evidence_mse_expanded(Phi, t, B, T):
    max_iteration = 100
    eps = 1e-16

    # Initial random alpha and beta
    [alpha,beta] = np.random.uniform(0, 10, [2,1])
    N = float(len(t))

    # Repeatedly used inner product and quantities
    phi_transpose_phi = np.dot(np.transpose(Phi), Phi)
    phi_transpose_t = np.dot(np.transpose(Phi), t)
    eigen_values, eigen_vectors = np.linalg.eig(phi_transpose_phi)

    #
    alpha_cur = alpha
    beta_cur = beta

    for i in range(max_iteration):
        m_n = compute_m_n(alpha, beta, eigen_values, eigen_vectors, phi_transpose_t)
        gamma = compute_gamma(alpha, beta, eigen_values)

        alpha = gamma/np.dot(m_n, m_n)
        beta = compute_beta(N, gamma, m_n, Phi, t)

        if abs(alpha - alpha_cur) < eps and abs(beta - beta_cur) < eps:
            break
        alpha_cur = alpha
        beta_cur = beta

    # Compute log evidence
    log_evidence = compute_log_evidence(alpha, beta, eigen_values, m_n, Phi, t)

    # Compute mean squared error
    N = float(len(T))
    r = np.dot(B, m_n) - T
    mse = 1/N * np.sum(np.power(r, 2))
    return mse, log_evidence


"""
Compute the log evidence function
M/2 ln(a) + N/2 ln(b) -E(m_n) - 1/2 ln|A| -N/2 ln(2pi)

where
E(m_n) = b/2 |t - Phi m_n |^2 + a/2 m_n^2
A = (alpha * I + beta Phi'Phi)
"""
def compute_log_evidence(alpha, beta, eigenvalues, m_n, phi, t):
    # M is the number of features?
    M = len(phi[0])
    # N is the number of data/
    N = len(t)
    determinant_A = np.product(alpha*np.ones([M,1]) + beta *eigenvalues)
    expected_mn = beta/2 * np.sum(np.power(t-np.dot(phi, m_n),2))+alpha/2*np.dot(m_n, m_n)
    log_evidence = M/2*np.log(alpha) + N/2*np.log(beta)\
                   + expected_mn - 1/2*np.log(determinant_A)\
                   - N/2*np.log(2*math.pi)

    # if log_evidence == -math.inf:
    #     print("Determinant: ", str(determinant_A))
    #     print(min(eigenvalues), max(eigenvalues))
    #     print("alpha: ", alpha)
    #     print("beta: ", beta)
    return log_evidence



