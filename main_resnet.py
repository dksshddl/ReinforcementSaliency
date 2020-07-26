import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D, Input, Flatten, TimeDistributed, Dense, LSTM, Bidirectional, Reshape, \
    MaxPooling2D, Dropout, BatchNormalization
import py360convert

from custom_env.envs import CustomEnv
from dataset import Sal360


def make_model():
    input_shape = (None, 224, 224, 3)
    state_in = Input(input_shape)
    o_in = Input((None, 2))
    x = ConvLSTM2D(256, 3, 3, padding='same', return_sequences=True)(state_in)
    x = ConvLSTM2D(128, 3, 3, padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(64, 3, 3, padding='same', return_sequences=True)(x)
    print(x)
    x = TimeDistributed(MaxPooling2D())(x)
    print(x)
    x = TimeDistributed(Flatten())(x)
    x = Reshape([-1, 1024])(x)
    print(x)
    concat = keras.layers.concatenate([x, o_in])
    x = Bidirectional(LSTM(128, return_sequences=True))(concat)
    # x = Bidirectional(LSTM(64, activation=tf.nn.leaky_relu)(x))
    x = Dense(2, activation='linear')(x)

    m = keras.models.Model(inputs=[state_in, o_in], outputs=x)
    m.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    m.summary()
    return m


def make_densenet_model():
    input_shape = (1, 5, 224, 224, 3)
    state_in = Input(batch_shape=input_shape)
    o_in = Input(batch_shape=(1, 1, 2))
    feature = keras.applications.DenseNet121(include_top=False, pooling="avg")
    x = TimeDistributed(feature)(state_in)
    x = Flatten()(x)
    x = Reshape([1, -1])(x)
    concat = keras.layers.concatenate([x, o_in])
    x = Bidirectional(LSTM(128, return_sequences=True, stateful=True, activation="tanh"))(concat)
    x = Dense(2, activation="sigmoid")(x)
    m = keras.models.Model(inputs=[state_in, o_in], outputs=x)
    adam = tf.keras.optimizers.Adam(1e-6)
    m.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    m.summary()
    return m


def make_densenet_model2():
    input_shape = (1, None, 224, 224, 3)
    input_shape2 = (1, None, 64, 64, 3)

    state_in = Input(batch_shape=input_shape)
    state_in2 = Input(batch_shape=input_shape2)
    o_in = Input(batch_shape=(1, None, 2))

    x = ConvLSTM2D(64, 3, 1, padding='same', return_sequences=True, stateful=True)(state_in)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, 3, 2, padding='same', return_sequences=True, stateful=True)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D()(x)
    x = ConvLSTM2D(32, 3, 1, padding='same', return_sequences=True, stateful=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(32, 3, 2, padding='same', return_sequences=True, stateful=True)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D()(x)
    x = TimeDistributed(Flatten())(x)

    x2 = ConvLSTM2D(16, 3, 1, padding='same', return_sequences=True, stateful=True)(state_in2)
    x2 = BatchNormalization()(x2)
    x2 = ConvLSTM2D(16, 3, 2, padding='same', return_sequences=True, stateful=True)(x2)
    x2 = BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling3D()(x2)
    x2 = ConvLSTM2D(8, 3, 1, padding='same', return_sequences=True, stateful=True)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ConvLSTM2D(8, 3, 2, padding='same', return_sequences=True, stateful=True)(x2)
    x2 = BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling3D()(x2)
    x2 = TimeDistributed(Flatten())(x2)

    concat = keras.layers.concatenate([x, x2, o_in])

    out = LSTM(256, stateful=True, return_sequences=True)(concat)
    out = LSTM(256, stateful=True, return_sequences=True)(out)

    out = TimeDistributed(Dense(128, activation="relu"))(out)
    out = TimeDistributed(Dense(128, activation="relu"))(out)
    out = TimeDistributed(Dense(2, activation="relu"))(out)

    model = Model([state_in, state_in2, o_in], out)
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy", "mse"])
    return model


def make_densenet_model3():
    input_shape = (None, 224, 224, 3)
    input_shape2 = (None, 64, 64, 3)

    state_in = Input(shape=input_shape, batch_size=1)
    state_in2 = Input(shape=input_shape2, batch_size=1)
    o_in = Input(shape=(None, 2), batch_size=1)

    feature = keras.applications.DenseNet121(include_top=False, pooling="avg")
    x = TimeDistributed(feature)(state_in)
    x.set_shape((1, None, 1024))

    feature = keras.applications.DenseNet121(include_top=False, pooling="max", weights=None)
    x2 = TimeDistributed(feature)(state_in2)
    x2.set_shape((1, None, 1024))

    concat = keras.layers.concatenate([x, x2, o_in])
    encode, state_h, state_c = LSTM(512, stateful=True, return_state=True)(concat)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 2), batch_size=1)  # past sequence
    decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # decoder_states = [state_h, state_c]
    #
    # attention_score = tf.keras.layers.dot([decoder_states, encoder_states])
    # attention_distribution = tf.keras.layers.Softmax()(attention_score)
    # attention_value = tf.keras.layers.multiply([attention_distribution, encoder_states])
    # attention_concat = tf.keras.layers.concatenate([attention_value, decoder_states])
    decoder_dense1 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense2 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense3 = TimeDistributed(Dense(2, activation='sigmoid'))
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense3(decoder_outputs)

    # decoder_dense = TimeDistributed(Dense(2, activation='linear'))
    # decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([state_in, state_in2, o_in, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model([state_in, state_in2, o_in], encoder_states)
    # define inference decoder
    decoder_state_input_h = Input((512,))
    decoder_state_input_c = Input((512,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_dense1 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense2 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense3 = TimeDistributed(Dense(2, activation='sigmoid'))
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense3(decoder_outputs)

    decoder_states = [state_h, state_c]

    # decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy", "mse"])
    return model, encoder_model, decoder_model


def read_whole_video(cap, fx=0.3, fy=0.3):
    video = []
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, dsize=(224, 224))  # 원래 이미지의 fx, fy배
            # cv2.imshow("test",frame)
            video.append(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            cap.release()
            break
    # cv2.destroyAllWindows()
    return video


def generator(data, model, segment=2, mode="train"):
    if mode == "train":
        video_list = os.listdir(data.train_video_path)
        path = data.train_video_path
        size = 45
    elif mode == "val":
        video_list = os.listdir(data.train_video_path)
        path = data.train_video_path
        size = 12
    elif mode == "test":
        video_list = os.listdir(data.test_video_path)
        path = data.test_video_path
        size = 57
    else:
        raise ValueError(f"mode must be [train, val and test] but got {mode}")

    length = size * len(video_list)

    # model.reset_states()
    global dataset, enc, dec
    window_size = 20 // segment
    while True:
        idx_list = list(range(length))
        for _ in range(length):
            idx = random.choice(idx_list)
            idx_list.remove(idx)
            x_data1 = []
            x_data2 = []
            sal_data = []
            video_index = idx // size
            trajectory_index = idx % size
            target_video = video_list[video_index]
            video = dataset[target_video.split(".")[0]]

            if mode == "train":
                trajectory = data.train[0][target_video]
            elif mode == "val":
                trajectory = data.validation[0][target_video]
            elif mode == "test":
                trajectory = data.test[0][target_video]
            else:
                raise NotImplementedError("mode error!")
            for i in range(len(trajectory[trajectory_index])):
                lat, lng, start_frame, end_frame = trajectory[trajectory_index][i][2], \
                                                   trajectory[trajectory_index][i][
                                                       1], int(trajectory[trajectory_index][i][5]), int(
                    trajectory[trajectory_index][i][6])
                x = video[start_frame - 1:end_frame]
                x_data1.append(x[2])
                x_data2.append([lng, lat])  # 100, 2
            x1 = x_data1
            x2 = x_data2
            # y1 = x_data2[5:]
            bb = list(map(toDegree, x2))
            _x1 = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in zip(x1, bb)]
            enc.reset_states()
            for i in range(18):
                # model.reset_states()
                x1 = np.array([x_data1[i*5:(i + 1) * 5]])  # 0 ~ 50
                x2 = np.array([_x1[i*5:(i + 1) * 5]])  # 0 ~ 50
                x3 = np.array([x_data2[i*5:(i + 1) * 5]])  # 45 ~50
                y = np.array([x_data2[(i + 1) * 5:(i + 2) * 5]])  # 50 ~ 55
                yield [x1, x2, x3, x3], y


class SaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, enc, dec):
        self.enc = enc
        self.dec = dec
        self.enc_dir = os.path.join('weights', "mae_ed_only_enc")
        self.dec_dir = os.path.join('weights', "mae_ed_only_dec")
        if not os.path.exists(self.enc_dir):
            os.mkdir(self.enc_dir)
            os.mkdir(self.dec_dir)

        self.best_loss = None

    def on_epoch_end(self, epoch, logs):
        if self.best_loss is None:
            self.best_loss = logs['val_loss']
            self.enc.save_weights(os.path.join(self.enc_dir, "weights.h5"))
            self.dec.save_weights(os.path.join(self.dec_dir, "weights.h5"))
        else:
            if self.best_loss > logs['val_loss']:
                self.best_loss = logs['val_loss']
                self.enc.save_weights(os.path.join(self.enc_dir, "weights.h5"))
                self.dec.save_weights(os.path.join(self.dec_dir, "weights.h5"))


class Sequence(tf.keras.utils.Sequence):
    def __init__(self, dataset, mode="train"):
        self.data = dataset
        self.mode = mode

        if mode == "train":
            self.size = 45
        elif mode == "val":
            self.size = 12
        elif mode == "test":
            self.size = 57
        else:
            raise ValueError(f"mode must be [train, val and test] but got {self.mode}")

        if mode == "train" or mode == "val":
            self.video_list = os.listdir(self.data.train_video_path)
            self.path = self.data.train_video_path
        elif mode == "test":
            self.video_list = os.listdir(self.data.test_video_path)
            self.path = self.data.test_video_path

    def __len__(self):
        return self.size * len(self.video_list)

    def __getitem__(self, idx):
        global model, dataset, enc
        # model.reset_states()
        x_data1 = []
        x_data2 = []
        y_data = []
        video_index = idx // self.size
        trajectory_index = idx % self.size
        target_video = self.video_list[video_index]

        video = dataset[target_video.split(".")[0]]
        if self.mode == "train":
            trajectory = self.data.train[0][target_video]
        elif self.mode == "val":
            trajectory = self.data.validation[0][target_video]
        elif self.mode == "test":
            trajectory = self.data.test[0][target_video]
        else:
            raise NotImplementedError("mode error!")

        for i in range(len(trajectory[trajectory_index])):
            lat, lng, start_frame, end_frame = trajectory[trajectory_index][i][2], trajectory[trajectory_index][i][
                1], int(trajectory[trajectory_index][i][5]), int(trajectory[trajectory_index][i][6])
            # next_lat, next_lng = trajectory[trajectory_index][i+1][2], trajectory[trajectory_index][i+1][1]
            x = video[start_frame - 1:end_frame]
            x = x[2]  # 100 , 224, 224, 3
            x_data1.append(x)
            x_data2.append([lng, lat])  # 100, 2
            # y_data.append([next_lat, next_lng])

        # x_data1 = np.reshape(x_data1, [10, -1, 224, 224, 3]) # 10, 10, 224, 224, 3
        # x_data2 = np.reshape(x_data2, [10, -1, 2])  # 10, 10, 2
        batch_x1, batch_x2, batch_x3, batch_y = [], [], [], []
        bb = list(map(toDegree, x_data2))
        _x1 = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in zip(x_data1, bb)]

        # for i in range(5, 9):
        #     batch_x1.append(x_data1[i*10 - 50:i*10])
        #     batch_x2.append(_x1[i*10 - 50:i*10])
        #     batch_x3.append(x_data2[i*10-50:i*10])
        #     batdh_x4.append(x_data2[i*10-5:i*10])
        #     batch_y.append(x_data2[i*10:i*10+5])

        # return [np.array([_x1]), np.array([_x_data2])], np.array([y_data])


def toDegree(x):
    return (np.array(x) * 2 - 1) * np.array([180, 90])


def scheduler(epoch):
    if epoch < 10:
        return 0.01
    else:
        return 0.01 * tf.math.exp(0.1 * (10 - epoch))


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

board_callback = tf.keras.callbacks.TensorBoard(
    log_dir='log/mae_ed_only', write_graph=True, write_images=False,
    update_freq='batch')

test_board_callback = tf.keras.callbacks.TensorBoard(
    log_dir='log/random_test', write_graph=True, write_images=False,
    update_freq='batch')

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join("weights", "mae_ed_only"), monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')

early_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)

data360 = Sal360()

model, enc, dec = make_densenet_model3()

# model = tf.keras.models.load_model(os.path.join("weights", "mae_ed_conv"))
# enc.load_weights(os.path.join("weights", "mae_ed_conv_enc", "weights.h5"))
# dec.load_weights(os.path.join("weights", "mae_ed_conv_dec", "weights.h5"))

# model.load_weights(os.path.join("weights", "random0612"))
# model = tf.keras.models.load_model(os.path.join("weights", "random0612"))
model.summary()
enc.summary()
dec.summary()
save_callback = SaveCallback(enc, dec)

# #segment= 2
video_path = os.listdir(os.path.join("video_data", "224x224"))
dataset = {}
# saliency_dataset = {}

for path in video_path:
    name = path.split(".")[0]
    tmp = np.load(os.path.join("video_data", "224x224", path))
    # tmp_saliecny = np.load(os.path.join("saliency_data", "224x224", path))
    dataset[name] = tmp / 255.
    # saliency_dataset[name] = tmp_saliecny

train_gen = generator(data360, model, mode="train", segment=2)
val_gen = generator(data360, model, mode="val", segment=2)
test_gen = generator(data360, model, mode="test", segment=2)
hist = model.fit(train_gen, steps_per_epoch=630 * 18, validation_data=val_gen, validation_steps=12 * 14 * 18,
                 validation_freq=2, epochs=1000,
                 callbacks=[board_callback, ckpt_callback, lr_callback, early_callback])

# gen = Sequence(data360, mode="train")
# val_gen = Sequence(data360, mode="val")
# test_gen = Sequence(data360, mode="test")
# hist = model.fit(gen, validation_data=val_gen, validation_freq=1, epochs=1000, shuffle=True, callbacks=[board_callback, ckpt_callback, lr_callback, save_callback, early_callback])

# # hist = model.fit(gen, validation_data=val_gen, validation_freq=2, epochs=1000, shuffle=True)

# results = model.evaluate(test_gen)
# print('test loss, test acc:', results)

# result = model.predict_generator(Sequence(data360, mode="test"))
# print(np.shape(result))
# np.save(os.path.join("data", "random0612_tanh.npy"), result)
