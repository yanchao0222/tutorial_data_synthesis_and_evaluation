import tensorflow as tf
import numpy as np
import time
import os
import pandas as pd
import argparse


def prob2onehot(prob):
    return tf.cast((tf.reduce_max(prob, axis=-1, keepdims=True) - prob) == 0, tf.float32)

class Generator(tf.keras.Model):
    def __init__(self, parameter_dict):
        super(Generator, self).__init__()
        self.G_DIMS = [parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['dimension']-parameter_dict['race_dimension']]
        self.dense_layers = [tf.keras.layers.Dense(dim) for dim in self.G_DIMS[:-1]]
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization(epsilon=1e-5) for _ in self.G_DIMS[:-1]]
        self.output_layer_code = tf.keras.layers.Dense(self.G_DIMS[-1], activation=tf.nn.sigmoid)
        self.output_layer_race = tf.keras.layers.Dense(parameter_dict['race_dimension'], activation=tf.nn.softmax)

    def call(self, x, training):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=training))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=training))
            x += h
        x = tf.concat((self.output_layer_race(x), self.output_layer_code(x)),axis=-1)
        return x

    def test(self, x):
        h = self.dense_layers[0](x)
        x = tf.nn.relu(self.batch_norm_layers[0](h, training=False))
        for i in range(1,len(self.G_DIMS[:-1])):
            h = self.dense_layers[i](x)
            h = tf.nn.relu(self.batch_norm_layers[i](h, training=False))
            x += h
        x = tf.concat((prob2onehot(self.output_layer_race(x)), self.output_layer_code(x)),axis=-1)
        return x

def gen(modeln, parameter_dict):
    checkpoint_directory = "training_checkpoints_emrwgan_" + modeln
    checkpoint_prefix = '/YOUR_LOCAL_PATH/GAN_training/' + checkpoint_directory + "/ckpt-"
    generator = Generator(parameter_dict)

    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, directory='/YOUR_LOCAL_PATH/GAN_training/' + checkpoint_directory, max_to_keep=50)
    
    if parameter_dict['load_checkpoint_number'] == 'best':
        status = checkpoint.restore(manager.latest_checkpoint)
    else:
        checkpoint.restore(checkpoint_prefix + parameter_dict['load_checkpoint_number']).expect_partial()
    @tf.function
    def g_step():
        z = tf.random.normal(shape=[100, parameter_dict['Z_DIM']])
        synthetic = generator.test(z)
        return synthetic
    data_df = pd.read_csv(parameter_dict['training_data_path'])
    data = np.array(data_df.values).astype('float32')
    pos = int(np.sum(data[:,parameter_dict['outcome_dimension']] == 1)*3)
    neg = int(np.sum(data[:,parameter_dict['outcome_dimension']] == 0)*3)
    syn_pos = []
    syn_neg = []
    while len(syn_pos) <= pos:
        tmp = g_step().numpy()
        syn_pos.extend(tmp[tmp[:,parameter_dict['outcome_dimension']] >= 0.5])
    while len(syn_neg) <= neg:
        tmp = g_step().numpy()
        syn_neg.extend(tmp[tmp[:,parameter_dict['outcome_dimension']] < 0.5])
    syn = np.array(syn_pos+syn_neg)
    print(syn.shape)

    col_list = list(data_df.columns)
    continuous_col_name_list = [col_list[col_ind] for col_ind in parameter_dict['continuous_feature_col_ind']]
    feature_range = np.load(parameter_dict['feature_range_path'], allow_pickle=True).item()
    for col_name, i in zip(continuous_col_name_list, parameter_dict['continuous_feature_col_ind']):
        xmin, xmax = feature_range[col_name][0], feature_range[col_name][1]
        syn[:, i] = (1 - syn[:, i])*xmin + syn[:,i]*xmax
    np.save('/YOUR_LOCAL_PATH/GAN_training/syn/emrwgan_model_'+modeln+'_ckpt_'+parameter_dict['load_checkpoint_number']+'.npy', syn)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--load_checkpoint', type=str)
    args = parser.parse_args()

    parameter_dict = {}
    parameter_dict['training_data_path'] = '/YOUR_LOCAL_PATH/preprocessed_training_data.csv'
    parameter_dict['feature_range_path'] = '/YOUR_LOCAL_PATH/min_max_log.npy'
    parameter_dict['continuous_feature_col_ind'] = [1456,1457,1458,1459]
    parameter_dict['batchsize'] = 4096
    parameter_dict['Z_DIM'] = 128
    parameter_dict['dimension'] = 1460
    parameter_dict['h_dimension'] = 384
    parameter_dict['race_dimension'] = 6
    parameter_dict['outcome_dimension'] = 6

    parameter_dict['load_checkpoint_number'] = args.load_checkpoint 

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    gen(args.model_id, parameter_dict)
