"""
Perform linear regression with regularization

Given some data matrix A and target t, the solution to the
least squares problem with regularization is

w = (lam * I + A'A)^(-1) A't

"""
import numpy as np


"""
Load the data matrix from data_file
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
Load the target (result) vector from file
"""
def load_target_vector(target_file):
    f = open(target_file, "r")
    t = []
    for line in f:
        t.append(float(line.strip()))
    f.close()
    return np.asarray(t)


"""
Find w from the least square solution given matrix Phi, result t and lambda L
w is given as the solution to 
w = (L * I + Phi'Phi)^-1 Phi't
"""
def find_regularized_weight(Phi, t, L):
    n = len(Phi[0])
    I = np.identity(n, dtype=float)
    # Find the inverse term
    inverse = np.linalg.inv(L * I + np.dot(np.transpose(Phi), Phi))
    # inverse * Phi' * t
    return np.dot(np.dot(inverse, np.transpose(Phi)), t)


"""
Find mean squared error based on w, B and result t
It is 1/N * sum{(Bw - t)^2}
"""
def calculate_mse(w, B, T):
    # Find Bw - T
    residual = np.dot(B, w) - T
    error_squared = np.power(residual, 2)
    # mean squared error
    return 1/len(T) * np.sum(error_squared)


"""
    test regularizer on specific data groups
    over specified range of regularizer parameter lambda
"""
def load_data_tensors(train_file, train_target, test_file, test_target):
    A = load_data_matrix(train_file)
    t = load_target_vector(train_target)
    B = load_data_matrix(test_file)
    T = load_target_vector(test_target)
    return A,t,B,T


"""
    Test linear regression with regularization over L array of possible lambda
    values
    This returns test_mse and training_mse
"""
def test_regularizer(train_file, train_result_file, test_file, test_result_file, L_array):
    # Load the matrix
    Phi = load_data_matrix(train_file)
    t = load_target_vector(train_result_file)
    B = load_data_matrix(test_file) # Test data
    T = load_target_vector(test_result_file)
    test_mse_array = []
    train_mse_array = []
    for l in L_array:
        # Find the regularized weight
        w = find_regularized_weight(Phi, t, l)
        # Calculate test set MSE
        train_mse = calculate_mse(w, Phi, t)
        test_mse = calculate_mse(w, B, T)

        train_mse_array.append(train_mse)
        test_mse_array.append(test_mse)

    return train_mse_array, test_mse_array


"""
    Generate random subset of train_data with train_target of size N 
"""
def generate_subset(train_data, train_target, N):
    # generate random permutation
    indices = np.array(np.random.permutation(len(train_data)))
    permuted_data = []
    for id in indices: # Get the permuted train data
        permuted_data.append(train_data[id])
    permuted_target = np.take(train_target, indices) # Take the same target
    return permuted_data[:N], permuted_target[:N] # Return the subsets


"""
    Compute the learning curve given the training and testing data
    Over the size of the training set and with specific lambda array
"""
def compute_learning_curve(train_data, train_target, test_data, test_target, size_array, l_array):
    Phi = load_data_matrix(train_data)
    t = load_target_vector(train_target)
    B = load_data_matrix(test_data)
    T = load_target_vector(test_target)

    mse_dict = {}
    for l in l_array:
        mse_dict[l] = []

    # Go through all the size
    for size in size_array:
        # Get a random subset of data
        # Try on all the lambda values
        for l in l_array:
            mse = 0.0
            # Take average for 10
            for i in range(10):
                # For 10 repetitions, generate random subset of the training data
                a_permuted, t_permuted = generate_subset(Phi, t, int(size))
                w = find_regularized_weight(a_permuted, t_permuted, l)
                mse += calculate_mse(w, B, T) # Find the MSE on the test set
            mse /= 10
            mse_dict[l].append(mse)

    return mse_dict

