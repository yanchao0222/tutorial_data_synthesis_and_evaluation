import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os
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

class Discriminator(tf.keras.Model):
    def __init__(self, parameter_dict):
        super(Discriminator, self).__init__()
        self.D_DIMS = [parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension'], parameter_dict['h_dimension']]
        self.dense_layers = [tf.keras.layers.Dense(dim, activation=tf.nn.relu) for dim in self.D_DIMS]
        self.layer_norm_layers = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in self.D_DIMS]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense_layers[0](x)
        x = self.layer_norm_layers[0](x)
        for i in range(1,len(self.D_DIMS)):
            h = self.dense_layers[i](x)
            h = self.layer_norm_layers[i](h)
            x += h
        x = self.output_layer(x)
        return x


def train(modeln, parameter_dict):
    checkpoint_directory = "training_checkpoints_emrwgan_"+modeln
    # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint_prefix = '/YOUR_LOCAL_PATH/GAN_training/' + checkpoint_directory + "/ckpt-"
    data = np.array(pd.read_csv(parameter_dict['training_data_path']).values).astype('float32')

    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,reshuffle_each_iteration=True).batch(parameter_dict['batchsize'], drop_remainder=True)

    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)

    generator = Generator(parameter_dict)
    discriminator = Discriminator(parameter_dict)

    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, directory='/YOUR_LOCAL_PATH/GAN_training/' + checkpoint_directory, max_to_keep=50)

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[parameter_dict['batchsize'], parameter_dict['Z_DIM']])

        epsilon = tf.random.uniform(
            shape=[parameter_dict['batchsize'], 1],
            minval=0.,
            maxval=1.)

        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, False)
            interpolate = real + epsilon * (synthetic - real)

            real_output = discriminator(real)
            fake_output = discriminator(synthetic)

            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch(interpolate)
                interpolate_output = discriminator(interpolate)
            w_grad = t.gradient(interpolate_output, interpolate)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

            disc_loss = 10 * gradient_penalty + w_distance

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[parameter_dict['batchsize'], parameter_dict['Z_DIM']])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z,True)

            fake_output = discriminator(synthetic)

            gen_loss = -tf.reduce_mean(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        disc_loss, w_distance = d_step(batch)
        g_step()
        return disc_loss, w_distance

    print('training start', flush=True)

    best_loss = 1000000.0
    for epoch in range(15000):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        step = 0.0
        for args in dataset_train:
            loss, w = train_step(args)
            total_loss += loss
            total_w += w
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
        if epoch % 10 == 0:
            print(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch), flush=True)
            if epoch > 100 and epoch % 50 == 0 and -total_loss / step <= best_loss and -total_loss / step > 0:
                best_loss = -total_loss / step
                manager.save(checkpoint_number=epoch)
                print('ckpt %d saved with loss %.6f' % (epoch, best_loss), flush=True)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str)
    parser.add_argument('--model_id', type=str)
    args = parser.parse_args()

    parameter_dict = {}
    parameter_dict['training_data_path'] = '/YOUR_LOCAL_PATH/normalized_training_data.csv'
    parameter_dict['feature_range_path'] = '/YOUR_LOCAL_PATH/min_max_log.npy'
    parameter_dict['continuous_feature_col_ind'] = [1456,1457,1458,1459]
    parameter_dict['batchsize'] = 4096
    parameter_dict['Z_DIM'] = 128
    parameter_dict['dimension'] = 1460
    parameter_dict['h_dimension'] = 384
    parameter_dict['race_dimension'] = 6


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
    
    train(args.model_id, parameter_dict)



