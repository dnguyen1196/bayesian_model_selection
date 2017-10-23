import sys
import os
import regularizer.regularized_linear_regression as rlg
import bayes_model_selector.model_selector as evidence
import visualize.drawer as drawer
import numpy as np

"""
    DATA FILES
"""
train_data_sets = ["train-100-10.csv", "train-100-100.csv", "train-1000-100.csv", "train-crime.csv",
                  "train-wine.csv"]
train_targets = ["trainR-100-10.csv", "trainR-100-100.csv", "trainR-1000-100.csv", "trainR-crime.csv",
                 "trainR-wine.csv"]

test_data_sets = ["test-100-10.csv", "test-100-100.csv", "test-1000-100.csv", "test-crime.csv",
                 "test-wine.csv"]
test_targets = ["testR-100-10.csv", "testR-100-100.csv", "testR-1000-100.csv", "testR-crime.csv",
                "testR-wine.csv"]


# Data folders and results folders
# data_folder = os.path.join(os.getcwd(), "data")
data_folder = os.getcwd()
results_folder = os.path.join(os.getcwd(), "results/")


def main(argv):
    # do_regularized_linear_regression()
    # find_learning_curve()
    # optimize_evidence()
    learn_parameter_model_order()


"""
    TASK 1
"""
def do_regularized_linear_regression():
    # Try all lambda values from 0 to 150
    L = range(151)
    save_folder = os.path.join(results_folder, "task1/")

    for i in range(len(train_data_sets)):
        train_data = os.path.join(data_folder, train_data_sets[i])
        train_target = os.path.join(data_folder, train_targets[i])
        test_data = os.path.join(data_folder, test_data_sets[i])
        test_target = os.path.join(data_folder, test_targets[i])

        # Find the result of test set mse against l_array
        train_mse_array, test_mse_array = rlg.test_regularizer(train_data, train_target, test_data, test_target, L)
        print("optimal lambda: ", np.argmin(test_mse_array))
        print("MSE: ", np.min(test_mse_array))

        # Plot test set mse against lambda
        drawer.plot_mse_against_l(L, train_mse_array, test_mse_array, train_data_sets[i], save_folder)


"""
    TASK 2
"""
def find_learning_curve():
    save_folder = os.path.join(results_folder, "task2/")
    # Generate 100 sizes between 10 and 800
    size_array = np.linspace(10, 800, 200)

    # Only test on 1000-100 dataset
    train_data = os.path.join(data_folder, "train-1000-100.csv")
    train_target = os.path.join(data_folder, "trainR-1000-100.csv")
    test_data = os.path.join(data_folder, "test-1000-100.csv")
    test_target = os.path.join(data_folder, "testR-1000-100.csv")
    l_array = [5,27,145] # try regularization parameter of values 5, 27 and 145

    mse_dict = rlg.compute_learning_curve(train_data, train_target, test_data,
                               test_target,size_array, l_array)

    drawer.plot_learning_curve(mse_dict, size_array, save_folder, "train-1000-100.csv")


"""
    TASK 3
"""
def optimize_evidence():
    mse_array = []
    save_folder = os.path.join(results_folder, "task3/")

    # Go through all the training size
    for i in range(len(train_data_sets)):
        train_file = os.path.join(data_folder, train_data_sets[i])
        train_r_file = os.path.join(data_folder, train_targets[i])
        test_file = os.path.join(data_folder, test_data_sets[i])
        test_r_file = os.path.join(data_folder, test_targets[i])

        mse = evidence.compute_optimal_evidence_mse(train_file, train_r_file, test_file, test_r_file)
        mse_array.append(mse)

    for i in range(len(mse_array)):
        print(train_data_sets[i], " : ", mse_array[i])
    drawer.plot_mse_for_evidence_maximizer(mse_array, train_data_sets, save_folder)

"""
    TASK 4
"""
def learn_parameter_model_order():
    train_datafiles = ["train-f3.csv", "train-f5.csv"]
    train_targetfiles = ["trainR-f3.csv", "trainR-f5.csv"]
    test_datafiles = ["test-f3.csv", "test-f5.csv"]
    test_targetfiles = ["testR-f3.csv", "testR-f5.csv"]
    n = 10

    save_folder = os.path.join(results_folder, "task4")

    # Go through 2 training data sets
    for i in range(len(train_datafiles)):
        train_file = os.path.join(data_folder, train_datafiles[i])
        train_r_file = os.path.join(data_folder, train_targetfiles[i])
        test_file = os.path.join(data_folder, test_datafiles[i])
        test_r_file = os.path.join(data_folder, test_targetfiles[i])

        file_name = train_datafiles[i].split(".")[0]
        save_file = os.path.join(save_folder, file_name +".png")

        phi = evidence.load_data_matrix(train_file)
        t = evidence.load_target_vector(train_r_file)
        B = evidence.load_data_matrix(test_file)
        T = evidence.load_target_vector(test_r_file)

        mse_unreg_array = []
        mse_array = []
        evidence_array = []
        print("Data: ", train_file)

        for d in range(1,n+1):
            # Obtain augmented feature space
            phi_augmented = evidence.expand_feature_space(phi, d)
            B_augmented = evidence.expand_feature_space(B, d)

            # Perform unregularized linear regression
            w_unregularized = rlg.find_regularized_weight(phi_augmented, t, 0)
            mse_unregularized = rlg.calculate_mse(w_unregularized, B_augmented, T)

            # Compute mse + log evidence
            mse_bayesian, log_evidence = evidence.compute_evidence_mse_expanded(phi_augmented, t, B_augmented, T)

            mse_unreg_array.append(mse_unregularized)
            mse_array.append(mse_bayesian)
            evidence_array.append(log_evidence)

            print("dimension: ", str(d))
            print("mse unregularized linear regression: ", mse_unregularized)
            print("mse model selection: ", mse_bayesian)
            print("log evidence: ", log_evidence)

        drawer.plot_mse_against_dimension(mse_array, mse_unreg_array, evidence_array, range(1,n+1), save_file)


if __name__ == "__main__":
    main(sys.argv[1:])