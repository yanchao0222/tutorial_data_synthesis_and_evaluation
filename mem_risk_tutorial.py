import pandas as pd
import numpy as np
import os.path
import time
import sys

'''
Usage: synthetic_risk_model_mem.py [model_id] [ckpt_id] [theta] [train_filename] [test_filename] [output_directory]
Example: mem_risk_tutorial.py 21 best 5 /YOUR_LOCAL_PATH/original_training_data.csv /YOUR_LOCAL_PATH/original_testing_data.csv result/
1. [model_id]: model id. Default: 'real'.
2. [ckpt_id]: checkpoint id of a generative model. Default: 'best'.
3. [theta]: the threshold for the euclidean distance between two records. Default: '5'. Try: '10' and '20'.
4. [train_filename]: the filename of the training file. 
5. [test_filename]: the filename of the test file. 
6. [output_directory]: output directory. Default: 'result/'.
'''

def find_replicant(real, fake):
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    b = np.dot(fake, real.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)

def each_group(model):
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)
    if model_id != 'real':
        steps = np.ceil(n_train / batchsize)
        for i in range(int(steps)):
            distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)

    steps = np.ceil(n_test / batchsize)
    for i in range(int(steps)):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)

    n_tp = np.sum(distance_train <= theta)  # true positive counts
    n_fn = n_train - n_tp
    n_fp = np.sum(distance_test <= theta)  # false positive counts
    f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  # F1 score
    return f1


if __name__ == '__main__':
    # Default configuration
    dataset = "mimic"
    model_id = "real"
    ckpt_id = "best"
    theta = 5
    original_patient_filename_train = '/YOUR_LOCAL_PATH/original_training_data.csv'
    original_patient_filename_test = '/YOUR_LOCAL_PATH/original_testing_data.csv'
    n_cont_col = 4  # number of columns for continuous features from the right
    Result_folder = "./result/"
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)
    batchsize = 1000
    exp_name = "Mem_Risk"

    start1 = time.time()
    # Enable the input of parameters
    if len(sys.argv) >= 2:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3:
        ckpt_id = sys.argv[2]
    if len(sys.argv) >= 4:
        theta = float(sys.argv[3])
    if len(sys.argv) >= 5:
        original_patient_filename_train = sys.argv[4]
    if len(sys.argv) >= 6:
        original_patient_filename_test = sys.argv[5]
    if len(sys.argv) >= 7:
        Result_folder = sys.argv[6]
    print("output_directory: " + Result_folder)
    print("train_filename: " + original_patient_filename_train)
    print("test_filename: " + original_patient_filename_test)
    print("theta: " + str(theta))

    # load datasets
    train_df = pd.read_csv(original_patient_filename_train)
    col_name_list = list(train_df.columns)
    train = train_df.values
    
    test_df = pd.read_csv(original_patient_filename_test)
    test = test_df.values

    n_train = np.shape(train)[0]
    n_test = np.shape(test)[0]
    if model_id == 'real':
        fake = train.copy()
    else:
        syn_data = np.load(f'/YOUR_LOCAL_PATH/GAN_training/syn/emrwgan_model_{model_id}_ckpt_{ckpt_id}.npy',allow_pickle=True)
        for i in range(len(train[0])-4):
            syn_data[:,i] = (syn_data[:,i] >= 0.5)*1.0
        syn_data_df = pd.DataFrame(syn_data, columns = col_name_list)
        positive_outcome_syn_data = syn_data_df[syn_data_df['DIE_1y'] == 1.0].values
        negative_outcome_syn_data = syn_data_df[syn_data_df['DIE_1y'] == 0.0].values
        syn_data_df = np.concatenate((positive_outcome_syn_data[:14243,:], negative_outcome_syn_data[:112279,:]), axis=0)  ## make sure the synthetic dataset has the same number of positive and negative records than those in 70% real data (ie, training dataset)
        syn_data_df = pd.DataFrame(syn_data_df, columns = col_name_list)
        fake = pd.DataFrame(syn_data_df, columns = col_name_list).values

    elapsed1 = (time.time() - start1)
    start2 = time.time()
    # normalization
    [n_row, n_col] = fake.shape
    for j in range(n_col-n_cont_col, n_col):
        normal_max = np.amax(fake[:, j])
        normal_min = np.amin(fake[:, j])
        normal_range = normal_max - normal_min
        fake[:, j] = (fake[:, j] - normal_min) / normal_range
        train[:, j] = (train[:, j] - normal_min) / normal_range
        test[:, j] = (test[:, j] - normal_min) / normal_range
    result = each_group(model_id)
    elapsed2 = (time.time() - start2)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1 + elapsed2) + " seconds.")
    print("Loading time used: " + str(elapsed1) + " seconds.")
    print("Computing time used: " + str(elapsed2) + " seconds.")
    with open(Result_folder + "_" + str(model_id) + 'ckpt_id' + str(ckpt_id) + "_theta" + str(theta) + ".txt", 'w') as f:
        f.write(str(result) + "\n")
        f.write("Time used: " + str(elapsed1 + elapsed2) + " seconds.\n")
        f.write("Loading time used: " + str(elapsed1) + " seconds.\n")
        f.write("Computing time used: " + str(elapsed2) + " seconds.\n")
