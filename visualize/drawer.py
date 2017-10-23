import matplotlib.pyplot as plt
import os


def plot_mse_against_l(l_array, train_mse, test_mse, data_name, save_folder):
    plt.plot(l_array, test_mse, label="Test set MSE against regularization parameter",
             marker='o', markersize=5)
    plt.plot(l_array, train_mse, label="Train set MSE against regularization parameter",
             marker='o', markersize=5)

    plt.xlabel('Lambda (regularization parameter)')
    plt.ylabel('MSE')
    plt.title('Variations of MSE with regularization parameter for ' + data_name)
    plt.legend(["Test set MSE", "Train set MSE"])
    file_name_splits = data_name.split(".")
    # Save results
    save_file = os.path.join(save_folder, file_name_splits[0] + ".png")
    plt.savefig(save_file)
    plt.close()


def plot_learning_curve(mse_dict, size_array, save_folder, save_name):
    lambda_array = []
    file_name_splits = save_name.split(".")
    save_file = os.path.join(save_folder, file_name_splits[0] + ".png")

    line_style = ['-', '--','-.']
    i = 0
    for l in mse_dict:
        mse_array = mse_dict[l]
        lambda_array.append(l)
        plt.plot(size_array, mse_array, linestyle=line_style[i])
        i += 1

    plt.xlabel('Training size')
    plt.ylabel('Test set MSE')
    plt.legend(["Lambda = " + str(l) for l in lambda_array])
    plt.savefig(save_file)
    plt.close()


def plot_mse_for_evidence_maximizer(mse_array, data_set_array, save_folder):
    n_groups = len(mse_array)
    color = ['r', 'b', 'g', 'y', 'black']
    for i in range(n_groups):
        plot = plt.bar(i+1, mse_array[i], color=color[i], label=data_set_array[i])
    plt.legend(data_set_array)
    plt.ylabel("Test set MSE")
    save_file = os.path.join(save_folder, "task_3.png")
    plt.savefig(save_file)


def plot_mse_against_dimension(mse_array, mse_unreg_array, log_evidence_array, d, save_file):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(d, mse_array, label="mse against d (Bayesian model)")
    ax1.plot(d, mse_unreg_array, label="mse against d (unregularized)")
    ax1.set_xlabel("d")
    ax1.set_ylabel("mse")

    ax2.plot(d, log_evidence_array, label="log evidence against d", color='black')
    ax2.set_ylabel("log evidence")

    ax1.legend(["Bayesian Model", "Unregularized linear regression"])
    ax2.legend(["Log evidence"])
    plt.savefig(save_file)