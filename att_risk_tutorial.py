import numpy as np
import time
import pandas as pd
import numpy as np
from scipy import stats
import os.path
import sys

'''
Usage: synthetic_risk_model_attr.py [model_id] [ckpt_id] [x] [y] [original_filename] [output_directory]
Example: att_risk_tutorial.py 21 best 0 8 /YOUR_LOCAL_PATH/original_training_data.csv result/
1. [model_id]: model id. Default: 'real'.
2. [ckpt_id]: checkpoint id of a generative model. Default: 'best'.
3. [x]: 10 to x is the number of neighbours. A integer larger than -1. Default: '0'. Try: '1'.
4. [y]: 2 to y is the number of sensitive attributes A integer larger than -1. Default: '8'. Try: '10'.
5. [original_filename]: the filename of the original patient file.
6. [output_directory]: output directory. Default: 'result/'.
'''

def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def find_neighbour(r, r_, data, data_, k, cont_sense_attr):
    # k: k nearest neighbours

    diff_array = np.abs(data - r)
    diff_array_max = np.amax(diff_array, axis=0)
    diff_array_max2 = np.maximum(diff_array_max, 1)
    diff_array_rate = diff_array/diff_array_max2
    diff = np.sum(diff_array_rate, axis=1)
    thresh = np.sort(diff)[k-1]
    idxs = np.arange(len(data))[diff <= thresh]  # not exactly k neighbours?
    predict = stats.mode(data_[idxs])[0][0]

    if N_cont > 0:
        bin_r_ = r_[np.logical_not(cont_sense_attr)]
        bin_predict = predict[np.logical_not(cont_sense_attr)]
        cont_r_ = r_[cont_sense_attr]
        cont_predict = predict[cont_sense_attr]
        bin_n = len(bin_r_)  # number of binary attributes
        true_pos = ((bin_predict + bin_r_) == 2)
        false_pos = np.array([(bin_r_[i] == 0) and (bin_predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(bin_r_[i] == 1) and (bin_predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = np.logical_and(cont_predict <= cont_r_ * 1.1, cont_predict >= cont_r_ * 0.9)
    else:
        bin_n = len(r_)  # number of binary attributes
        true_pos = ((predict + r_) == 2)
        false_pos = np.array([(r_[i] == 0) and (predict[i] == 1) for i in range(bin_n)])
        false_neg = np.array([(r_[i] == 1) and (predict[i] == 0) for i in range(bin_n)])
        correct_cont_predict = 0
    return true_pos, false_pos, false_neg, correct_cont_predict

class Model(object):
    def __init__(self, fake, n, k, attr_idx):
        self.fake = fake
        self.n = n  # number of attributes used by the attacker
        self.k = k  # k nearest neighbours
        self.true_pos = []
        self.false_pos = []
        self.false_neg = []
        self.attr_idx = attr_idx  # selected attributes' indexes
        self.attr_idx_ = np.array([j for j in range(N_attr) if j not in attr_idx])  # unselected attributes' indexes
        self.data = self.fake[:, self.attr_idx]
        self.data_ = self.fake[:, self.attr_idx_]
        if N_cont > 0:
            self.correct = []
            self.cont_sense_attr = cont_sense[self.attr_idx_]

    def single_r(self, R):
        r = R[self.attr_idx]  # tested record's selected attributes
        r_ = R[self.attr_idx_]  # tested record's unselected attributes
        if N_cont > 0:
            true_pos, false_pos, false_neg, correct = find_neighbour(r, r_, self.data, self.data_, self.k, self.cont_sense_attr)
            self.correct.append(correct)
        else:
            true_pos, false_pos, false_neg, _ = find_neighbour(r, r_, self.data, self.data_, self.k, 0)
        self.true_pos.append(true_pos)
        self.false_pos.append(false_pos)
        self.false_neg.append(false_neg)

def cal_score(n, k):
    # 2^n: the number of attributes used by the attacker
    # 10^k: the number of neighbours

    real_disease = real[:, SENSE_BEGIN:SENSE_END]
    disease_attr_idx = np.flipud(np.argsort(np.mean(real_disease, axis=0)))[:2**n]  # sorted by how common a disease is

    attr_idx = np.concatenate([np.array(range(SENSE_BEGIN)), np.array([N_attr - 4]), disease_attr_idx + SENSE_BEGIN]) # move age to the demo block
    model = Model(fake, 2 ** n, 10 ** k, attr_idx)
    n_rows = np.shape(real)[0]
    for i in range(n_rows):
        if i % 100 == 0:
            print("patient#: " + str(i))
        record = real[i, :]
        model.single_r(record)

    # binary part
    tp_array = np.stack(model.true_pos, axis=0)  # array of true positives
    fp_array = np.stack(model.false_pos, axis=0)  # array of false positives
    fn_array = np.stack(model.false_neg, axis=0)  # array of false negatives
    tpc = np.sum(tp_array, axis=0)  # vector of true positive count
    fpc = np.sum(fp_array, axis=0)  # vector of false positive count
    fnc = np.sum(fn_array, axis=0)  # vector of false negative count
    f1 = np.nan_to_num(tpc / (tpc + 0.5 * (fpc + fnc)))

    # continuous part
    if N_cont > 0:
        correct_array = np.stack(model.correct, axis=0)  # array of correctness
        accuracy = np.mean(correct_array, axis=0)

    # compute weights
    entropy = []
    real_ = real[:, model.attr_idx_]
    n_attr_ = np.shape(real_)[1]  # number of predicted attributes
    for j in range(n_attr_):
        entropy.append(get_entropy(real_[:, j]))
    weight = np.asarray(entropy) / sum(entropy)
    if N_cont > 0:
        bin_weight = weight[np.logical_not(model.cont_sense_attr)]
        cont_weight = weight[model.cont_sense_attr]
        score = np.sum(np.concatenate([f1, accuracy]) * np.concatenate([bin_weight, cont_weight]))
    else:
        score = np.sum(f1 * weight)
    return score


if __name__ == '__main__':
    # Default configuration
    dataset = "mimic" 
    model_id = "real"
    ckpt_id = "best"
    x = 0  # 10 to x is the number of neighbours [0, 1]
    y = 8  # 2 to y is the number of sensitive attributes used by the attacker [0, 11]
    original_patient_filename = '/YOUR_LOCAL_PATH/original_training_data.csv'
    Result_folder = "./result/"

    start1 = time.time()
    # Enable the input of parameters
    if len(sys.argv) >= 2:
        model_id = sys.argv[1]
    if len(sys.argv) >= 3:
        ckpt_id = sys.argv[2]
    if len(sys.argv) >= 4:
        x = int(sys.argv[3])
    if len(sys.argv) >= 5:
        y = int(sys.argv[4])
    if len(sys.argv) >= 6:
        original_patient_filename = sys.argv[5]
    if len(sys.argv) >= 7:
        Result_folder = sys.argv[6]
    print("output_directory: " + Result_folder)
    print("original_filename: " + original_patient_filename)
    print("syn model_id: " +  model_id)
    print("x: " + str(x))
    print("y: " + str(y))
    if not os.path.exists(Result_folder):
        os.mkdir(Result_folder)

    SENSE_BEGIN = 8  # first 8 attributes are not sensitive
    N_attr = 1460  # number of total attributes
    N_cont = 4  # number of continuous attributes

    SENSE_END = N_attr - N_cont
    if N_cont > 0:
        cont_sense = np.array([False for i in range(SENSE_END)] + [True for i in range(N_cont)])
    exp_name = "Attr_Risk"

    # load datasets
    real_df = pd.read_csv(original_patient_filename)
    col_name_list = list(real_df.columns)
    real = real_df.values

    if model_id == 'real':
        fake = real
    else:
        syn_data = np.load(f'/YOUR_LOCAL_PATH/GAN_training/syn/emrwgan_model_{model_id}_ckpt_{ckpt_id}.npy',allow_pickle=True)
        for i in range(len(real[0])-4):
            syn_data[:,i] = (syn_data[:,i] >= 0.5)*1.0
        syn_data_df = pd.DataFrame(syn_data, columns = col_name_list)
        positive_outcome_syn_data = syn_data_df[syn_data_df['DIE_1y'] == 1.0].values
        negative_outcome_syn_data = syn_data_df[syn_data_df['DIE_1y'] == 0.0].values
        syn_data_df = np.concatenate((positive_outcome_syn_data[:14243,:], negative_outcome_syn_data[:112279,:]), axis=0)  ## make sure the synthetic dataset has the same positive and negative records as those in the 70% real data (ie, the training dataset).
        syn_data_df = pd.DataFrame(syn_data_df, columns = col_name_list)
        fake = pd.DataFrame(syn_data_df, columns = col_name_list).values
        
    result = cal_score(y, x)
    elapsed1 = (time.time() - start1)
    print("Risk: " + str(result) + ".")
    print("Time used: " + str(elapsed1) + " seconds.")
    with open(Result_folder + "_" + str(model_id) + 'ckpt_id' + str(ckpt_id) + "_x" + str(x) + "_y" + str(y) + ".txt", 'w') as f:
        f.write(str(result) + "\n")
        f.write("Time used: " + str(elapsed1) + " seconds.\n")
