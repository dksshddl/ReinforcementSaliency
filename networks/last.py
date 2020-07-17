import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, concatenate, Conv2D, MaxPooling2D, Flatten, BatchNormalization, \
    Reshape, UpSampling2D
from tensorflow.keras.models import Model, Sequential
import cv2

from dataset import Sal360, read_whole_video
from utils.binary import read_SalMap
from utils.config import video_path

from custom_env.envs import CustomEnv


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024, activation="tanh"))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', activation='tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='tanh', padding='same', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def saliecny_model():
    inputs = Input(batch_shape=(1, 5, 224, 224, 3))
    # attention = Input((None, 1024))
    feature = tf.keras.applications.DenseNet121(include_top=False, pooling='max')

    feature.summary()
    x = tf.keras.layers.TimeDistributed(feature)(inputs)
    # concat = tf.keras.layers.multiply([x, attention])
    lstm = LSTM(256, stateful=True)(x)
    out = Dense(16, activation='sigmoid')(lstm)
    m = Model(inputs=inputs, outputs=out)
    m.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    m.summary()
    return m


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return 100

    def __getitem__(self, item):
        return np.array([self.x[item]]), np.array([self.y[item]])


if __name__ == '__main__':
    model = saliecny_model()
    binary_loss = tf.keras.losses.binary_crossentropy
    opt = tf.keras.optimizers.Adam(2e-6)
    dataset = Sal360()
    train_path = os.path.join(video_path, "train", "320x160")
    test_path = os.path.join(video_path, "test", "320x160")
    train_videos = sorted(os.listdir(train_path))
    test_videos = sorted(os.listdir(test_path))
    writer_path = os.path.join("log", "test")
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    writer = tf.summary.create_file_writer(writer_path)
    writer.set_as_default()
    env = CustomEnv()
    checkpoint_directory = os.path.join("weights", "test")
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model, )
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join("log", "test"), update_freq='epoch')

    @tf.function
    def compute_grad(oo, tt):
        with tf.GradientTape() as tape:
            aa = model(oo, training=True)
            losses = binary_loss(tt, aa)
        grad = tape.gradient(losses, model.trainable_weights)
        return grad, losses


    step = 0
    epochs = 0
    MAX_EPOCHS = 30_000
    while epochs < MAX_EPOCHS:

        ob, saliency, target_video, tile = env.reset(trajectory=True, fx=1, fy=1, saliency=True, inference=False)
        obs = []
        saliencys = []
        tiles = []
        print(f"start {target_video}")
        while True:
            next_ob, next_saliency, done, next_tile = env.step([0, 0])
            # saliency_mean = np.mean(saliency)
            # saliency_std = np.std(saliency)
            # saliency = [(x - saliency_mean) / (saliency_std + 1e-7) for x in saliency]
            # saliency = np.reshape(saliency, [-1, 32 * 32])
            o = tf.keras.applications.densenet.preprocess_input(np.array(ob))
            o = o[:5, :, :, :]
            t = tf.convert_to_tensor(np.reshape(tile, [-1]))
            # saliency = tf.convert_to_tensor([saliency])
            obs.append(o)
            saliencys.append(saliency)
            tiles.append(t)
            o = tf.convert_to_tensor([o])
            # grad, losses = compute_grad(o, t)
            # tf.summary.scalar("loss", float(losses), step=step)
            # opt.apply_gradients(zip(grad, model.trainable_weights))

            if done:
                o = tf.keras.applications.densenet.preprocess_input(np.array(next_ob))
                o = o[:5, :, :, :]
                t = tf.convert_to_tensor(np.reshape(next_tile, [-1]))

                obs.append(o)
                saliencys.append(next_saliency)
                tiles.append(t)
                print(np.shape(obs), np.shape(saliencys), np.shape(tiles))
                model.fit(DataGenerator(obs, tiles), epochs=1, callbacks=[tensorboard_callback])
                model.reset_states()
                break
            step += 1
            ob = next_ob
            saliency = next_saliency
            tile = next_tile
        if epochs % 25 == 0:
            model.save(os.path.join(checkpoint_directory, "test.h5"), overwrite=True)

        epochs += 1
