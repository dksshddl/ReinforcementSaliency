import os
import time
from collections import deque
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import ConvLSTM2D, Input, Flatten, TimeDistributed, Dense, LSTM, Bidirectional, Reshape, \
    MaxPooling2D, BatchNormalization
from custom_env.envs import CustomEnv
from dataset import Sal360
import csv
from utils.config import data_path, scanpaths_H_path, fps, saliency_map_H_path
from utils.binary import get_SalMap_info, read_SalMap
from custom_env.envs import CustomEnv
from utils.equirectangle import Equirectangular, NFOV
import py360convert
from math import pi
import matplotlib.pyplot as plt
from tratra import calOpticalFlow

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
    x = Dense(2, activation="linear")(x)
    m = keras.models.Model(inputs=[state_in, o_in], outputs=x)
    adam = tf.keras.optimizers.Adam(1e-6)
    m.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    m.summary()
    return m


def make_densenet_model2():
    input_shape = (None, 224, 224, 3)
    input_shape2 = (None, 64, 64, 3)

    state_in = Input(shape=input_shape, batch_size=1)
    state_in2 = Input(shape=input_shape2, batch_size=1)
    # o_in = Input(batch_shape=(1, 5, 2))

    feature = keras.applications.MobileNetV2(include_top=False, pooling="avg")
    x = TimeDistributed(feature)(state_in)
    x.set_shape((1, None, 1024))

    feature2 = keras.applications.MobileNetV2(include_top=False, pooling="avg")
    x2 = TimeDistributed(feature2)(state_in)
    x2.set_shape((1, None, 1024))

    concat = keras.layers.Add([0.3 * x, 0.7 * x2])
    encode, state_h, state_c = LSTM(512, stateful=True, return_state=True)(concat)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 2), batch_size=1)  # o_in
    decoder_lstm = LSTM(512, return_sequences=True, stateful=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(2, activation='linear'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([state_in, state_in2, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = tf.keras.models.Model([state_in, state_in2], encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(batch_shape=(1, 512))
    decoder_state_input_c = Input(batch_shape=(1, 512))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs])
    # return all models
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy", "mse"])
    return model, encoder_model, decoder_model


class VideoChangeCallback(tf.keras.callbacks.Callback):
    def __init__(self, gener, dataset):
        super().__init__()
        self.gen = gener
        self.data = dataset

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        if self.gen.mode == "train":
            video, trajectory = self.data.get_data(trajectory=True, saliency=False, inference=False, target_video=None)
            video = [cv2.resize(f, dsize=(224, 224)) for f in video]
        elif self.gen.mode == "val":
            video, trajectory = self.data.get_data(trajectory=True, saliency=False, inference=False, target_video=None)
            video = [cv2.resize(f, dsize=(224, 224)) for f in video]
        elif self.gen.mode == "test":
            video, trajectory = self.data.get_data(trajectory=True, saliency=False, inference=False, target_video=None)
            video = [cv2.resize(f, dsize=(224, 224)) for f in video]
        else:
            raise ValueError(f"mode must be train, validation and test but got {self.gen.mode}")
        self.gen.x = video
        self.gen.y = trajectory


class Sequence(tf.keras.utils.Sequence):
    def __init__(self, dataset, model, videoset, mode="train", idx=None, target=None):
        self.model = model
        self.data = dataset
        self.mode = mode
        self.target = target
        self.videoset = videoset
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
            if idx is not None:
                self.video_list = [self.video_list[idx]]
                print(self.video_list)
            if target is not None:
                self.video_list = [self.target]
            self.path = self.data.test_video_path

    def __len__(self):
        # return self.size * len(self.video_list)
        return 57

    def __getitem__(self, idx):
        x_data1 = []
        x_data2 = []
        y_data = []
        video_index = idx // self.size
        trajectory_index = idx % self.size
        target_video = self.video_list[video_index]

        video = self.videoset[target_video.split(".")[0]]
        if self.mode == "train":
            trajectory = self.data.train[0][target_video]
        elif self.mode == "val":
            trajectory = self.data.validation[0][target_video]
        elif self.mode == "test":
            trajectory = self.data.test[0][target_video]
        else:
            raise NotImplementedError("mode error!")

        self.model.reset_states()
        for i in range(len(trajectory[trajectory_index]) - 1):
            lat, lng, start_frame, end_frame = trajectory[trajectory_index][i][2], trajectory[trajectory_index][i][
                1], int(trajectory[trajectory_index][i][5]), int(trajectory[trajectory_index][i][6])
            next_lat, next_lng = trajectory[trajectory_index][i + 1][2], trajectory[trajectory_index][i + 1][1]
            x = video[start_frame - 1:end_frame]
            x = x[2]
            x_data1.append(x)
            x_data2.append([lat, lng])
            y_data.append([next_lat, next_lng])
            os.path.exists()
        return [np.array([x_data1]), np.array([x_data2])], np.array([y_data])


def load_dataset():
    video_path = os.listdir("video_data/448x224")
    dataset = {}
    # saliency_dataset = {}

    for path in video_path:
        name = path.split(".")[0]
        tmp = np.load(os.path.join("video_data/448x224", path))
        # tmp_saliecny = np.load(os.path.join("saliency_data", "32x32", path))
        # dataset[name] = tf.keras.applications.densenet.preprocess_input(tmp)
        dataset[name] = tmp
        # saliency_dataset[name] = tmp_saliecny
    return dataset


def train(weight_path=None, use_sequence=True):
    if weight_path is None:
        model = make_model()
    else:
        model = tf.keras.models.load_model(weight_path)

    def scheduler(epoch):
        if epoch < 1:
            return 0.0001
        else:
            return 0.0001 * tf.math.exp(0.1 * (1 - epoch))

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("weights", "random"), monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch')

    board_callback = tf.keras.callbacks.TensorBoard(
        log_dir='log/dense3', write_graph=True, write_images=False,
        update_freq='batch')

    dataset = load_dataset()

    if use_sequence:
        data360 = None
        train_gen = Sequence(data360, model, dataset, mode="train")
        val_gen = Sequence(data360, model, dataset, mode="val")
        test_gen = Sequence(data360, model, dataset, mode="test")
    else:
        data360 = None

        train_gen = generator(data360, model, dataset, mode="train", segment=2)
        val_gen = generator(data360, model, dataset, mode="val", segment=2)
        test_gen = generator(data360, model, dataset, mode="test", segment=2)

    hist = model.fit(train_gen, validation_data=val_gen, validation_freq=2, epochs=1000, shuffle=use_sequence,
                     callbacks=[board_callback, ckpt_callback, lr_callback])
    results = model.evaluate(test_gen)
    print('test loss, test acc:', results)


def generator(data, model, videoset, segment=2, mode="train"):
    if mode == "train":
        video_list = os.listdir(data.train_video_path)
        size = 45
    elif mode == "val":
        video_list = os.listdir(data.train_video_path)
        size = 12
    elif mode == "test":
        video_list = os.listdir(data.test_video_path)
        path = data.test_video_path
        size = 57
    else:
        raise ValueError(f"mode must be [train, val and test] but got {mode}")

    length = size * len(video_list)

    # global model
    # model.reset_states()

    while True:
        print("generator!")
        idx_list = list(range(length))
        for _ in range(length):
            idx = random.choice(idx_list)
            idx_list.remove(idx)
            x_data1 = []
            x_data2 = []
            video_index = idx // size
            trajectory_index = idx % size
            target_video = video_list[video_index]
            video = videoset[target_video.split(".")[0]]
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
            model.reset_states()
            x1 = x_data1[:-10]  # 90
            x2 = x_data2[:-10]  # 90
            y = x_data2[10:]  # 90
            x1 = np.reshape(x1, [-1, 10, 224, 224, 3])
            x2 = np.reshape(x2, [-1, 10, 2])
            y = np.reshape(y, [-1, 10, 2])
            for a, b, c in zip(x1, x2, y):
                yield [np.array([a]), np.array([b])], np.array([c])


def hello_test():
    a = Sal360()
    test_dir = os.listdir(a.test_video_path)
    model = tf.keras.models.load_model(os.path.join("weights", "convlstm3"))

    for video in test_dir:
        dataset = a.test[0][video]
        for trajectory in dataset:
            t_in = trajectory[:-5]
            t_out = trajectory[5:]
            pred_out = model(t_in)


def restore_model():
    input_shape = (None, 32, 32)
    input_shape2 = (None, 64, 64, 3)

    state_in = Input(shape=input_shape, batch_size=1)
    state_in2 = Input(shape=input_shape2, batch_size=1)
    o_in = Input(shape=(None, 2), batch_size=1)
    x = TimeDistributed(Flatten())(state_in)

    feature = keras.applications.MobileNetV2(include_top=False, pooling="avg", weights=None)
    x2 = TimeDistributed(feature)(state_in2)
    # x2.set_shape((1, None, 1024))

    concat = keras.layers.concatenate([x, x2, o_in])

    encode, state_h, state_c = LSTM(512, return_state=True)(concat)
    encoder_states1 = [state_h, state_c]
    # encode, state_h, state_c = LSTM(512, stateful=True, return_state=True)(concat)
    # encoder_states2 = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 2), batch_size=1)  # o_in
    decoder_lstm1 = LSTM(512, return_sequences=True, return_state=True)
    # decoder_lstm2 = LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states1)
    # decoder_outputs, _, _ = decoder_lstm2(decoder_inputs, initial_state=encoder_states1)
    # decoder_states = [state_h, state_c]
    #
    # attention_score = tf.keras.layers.dot([decoder_states, encoder_states])
    # attention_distribution = tf.keras.layers.Softmax()(attention_score)
    # attention_value = tf.keras.layers.multiply([attention_distribution, encoder_states])
    # attention_concat = tf.keras.layers.concatenate([attention_value, decoder_states])
    decoder_dense1 = TimeDistributed(Dense(128, activation=tf.nn.leaky_relu))
    decoder_dense2 = TimeDistributed(Dense(128, activation=tf.nn.leaky_relu))
    decoder_dense3 = TimeDistributed(Dense(2, activation='linear'))
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense3(decoder_outputs)

    # decoder_dense = TimeDistributed(Dense(2, activation='linear'))
    # decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([state_in, state_in2, o_in, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model([state_in, state_in2, o_in], encoder_states1)
    # define inference decoder
    decoder_state_input_h1 = Input(batch_shape=(None, 512))
    decoder_state_input_c1 = Input(batch_shape=(None, 512))
    # decoder_state_input_h2 = Input(batch_shape=(None, 512))
    # decoder_state_input_c2 = Input(batch_shape=(None, 512))
    decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
    # decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]
    decoder_outputs, state_h, state_c = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
    decoder_states1 = [state_h, state_c]
    # decoder_outputs, state_h, state_c = decoder_lstm2(decoder_inputs, initial_state=decoder_states_inputs1)
    # decoder_states2 = [state_h, state_c]
    decoder_dense1 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense2 = TimeDistributed(Dense(128, activation='relu'))
    decoder_dense3 = TimeDistributed(Dense(2, activation='linear'))
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    decoder_outputs = decoder_dense3(decoder_outputs)

    # decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs1,
                          [decoder_outputs] + decoder_states1)
    # return all models
    model.compile(optimizer="adam", loss="mae", metrics=["accuracy", "mse"])
    return encoder_model, decoder_model


def hello(max_epochs=57):
    data360 = Sal360()
    env = CustomEnv()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # model = tf.keras.models.load_model(os.path.join("weights", "convlstm3"))
    # model.summary()
    # enc, dec = restore_model()
    # enc.load_weights(os.path.join("weights", "mae_ed_mobile_enc", "weights.h5"))
    # dec.load_weights(os.path.join("weights", "mae_ed_mobile_dec", "weights.h5"))

    # enc = tf.keras.models.load_model(os.path.join("weights", "mae_ed_enc"))
    # dec = tf.keras.models.load_model(os.path.join("weights", "mae_ed_dec"))
    # enc.summary()
    # dec.summary()
    model = tf.keras.models.load_model(os.path.join("weights", "random0612"))
    model.summary()
    # exit()
    for ds in range(5):

        epoch = 0
        target = os.listdir(data360.test_video_path)

        target_v = target[ds]
        target_n = target_v.split(".")[0]
        # aa = os.path.join("data", f"random_{target_n}.npy")
        # loaded = np.load(aa)  # 57 99 2

        # tf.keras.utils.plot_model(model, show_shapes=True)

        print(target_v)
        f1 = []
        re = []
        pr = []
        acu = []

        while epoch < max_epochs:
            ob, ac, target_video = env.reset(trajectory=True, inference=True, saliency=False, randomness=False,
                                             target_video=target_v, video_type="test")
            # enc.reset_states()
            # dec.reset_states()
            model.reset_states()
            writer = cv2.VideoWriter(os.path.join("results", f"{target_n}_{epoch}.mp4"),
                                     fourcc, fps[target_video], (3840, 1920))
            # predicted = loaded[epoch]
            index = 0
            x1, x2 = [], []
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            pred_data = []
            true_data = [ac]
            pred_buffer = deque([None for _ in range(10)], maxlen=10)
            while True:
                _tp = 0
                _tn = 0
                _fp = 0
                _fn = 0
                ob = [cv2.resize(f, dsize=(224, 224)) for f in ob]
                x1.append(ob[2])
                x2.append(ac)
                predict = pred_buffer.popleft()
                next_ob, reward, done, next_ac = env.step(predict)
                true_data.append(next_ac)

                if len(x2) == 10:
                    # degree = list(map(toDegree, x2))
                    # field_of_view = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in
                    #                  zip(x1, degree)]
                    pred_val = model([np.array([x1]), np.array([x2])])

                    # state = enc([np.array([x1]), np.array([field_of_view])])
                    # pred_val, _, _ = dec([np.array([x2])] + state)
                    pred_val = np.reshape(pred_val, [-1, 2])
                    pred_data.append(pred_val)
                    # plt.plot(x2, 'r')
                    # plt.plot(pred_val, 'b')
                    # plt.show()
                    x1.clear()
                    x2.clear()
                    # x1.pop(0)
                    # x2.pop(0)
                    pred_buffer = deque(pred_val, maxlen=10)
                    # next_ob, reward, done, next_ac = env.step(pred_val[-1])

                # next_ac = pred_val[-1]
                # else:
                #     next_ob, reward, done, next_ac = env.step(None)
                env.render(writer=writer)
                # calculate metric
                if env.inference_view.active:
                    a = env.inference_view.tile_info()
                    b = env.view.tile_info()
                    for true in b:
                        if true in a:
                            _tp += 1
                        else:
                            _fn += 1
                    for pred in a:
                        if pred not in b:
                            _fp += 1
                    _tn += 25 - (_tp + _fn + _fp)

                    tp += _tp
                    tn += _tn
                    fp += _fp
                    fn += _fn
                if done:
                    writer.release()
                    break
                index += 1
                ob = next_ob
                ac = next_ac

            epoch += 1
            true_data = np.array(true_data)
            pred_data = np.array(pred_data)
            # l = list(map(lambda x : x *5, l))
            name = target_video.split(".")[0]
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 2, 1)
            # ax2 = fig.add_subplot(1, 2, 2)
            #
            # ax1.set_title(f"{name} - lng")
            # ax1.set_xlabel("timestep")
            # ax1.set_ylabel("longitude")
            # ax1.set_ylim(0, 1)
            # # ax1.scatter(l, true_data[:, 0], c='blue', label="true lng")
            # # ax1.scatter(l, pred_data[:, 0], c='red', label="pred lng")
            # ax1.plot(true_data[:, 0], c='blue', label="true lng")
            # ax1.plot(pred_data[:, 0], c='red', label="pred lng")
            # ax1.legend(loc="lower right")
            #
            # ax2.set_title(f"{name} - lat")
            # ax2.set_xlabel("timestep")
            # ax2.set_ylabel("latitude")
            # ax2.set_ylim(0, 1)
            # ax2.plot(true_data[:, 1], c='blue', label="true lat")
            # ax2.plot(pred_data[:, 1], c='red', label="pred lat")
            # # ax2.scatter(l, true_data[:, 1], c='blue', label="true lat")
            # # ax2.scatter(l, pred_data[:, 1], c='red', label="pred lat")
            # ax2.legend(loc="lower right")
            #
            # plt.show()

            # print(f"error: {np.square(true_data - pred_data).mean()}")

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            f1_score = 2 * ((precision * recall) / (precision + recall))
            f1.append(f1_score)
            pr.append(precision)
            re.append(recall)
            acu.append(acc)
            print(f"precision, recall, acc, f1-score: {precision}, {recall}, {acc}, {f1_score}")
        print(f"mean precision, recall, acc, f1-score : {np.mean(pr)}, {np.mean(re)}, {np.mean(acu)}, {np.mean(f1)}")


def makesalmap():
    data360 = None
    test = os.listdir(data360.test_video_path)
    train = os.listdir(data360.train_video_path)
    infos = get_SalMap_info()

    for video in train:
        info = infos[video]
        saliency = read_SalMap(info)
        saliency1 = [cv2.resize(f, dsize=(224, 224)) for f in saliency]
        saliency2 = [cv2.resize(f, dsize=(32, 32)) for f in saliency]
        for s, s1, s2 in zip(saliency, saliency1, saliency2):
            cv2.imshow("test1", s)
            cv2.imshow("test2", s1)
            cv2.imshow("test3", s2)
            cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    #     name = video.split(".")[0]
    #     np.save(os.path.join("saliency_data", f"{name}_32x32.npy"), np.array(saliency))
    # for video in test:
    #     info = infos[video]
    #     saliency = read_SalMap(info)
    #     saliency = [cv2.resize(f, dsize=(32, 32)) for f in saliency]
    #     name = video.split(".")[0]
    #     np.save(os.path.join("saliency_data", f"{name}_32x32.npy"), np.array(saliency))


def makedd():
    data360 = None

    test = os.listdir(data360.test_video_path)
    train = os.listdir(data360.train_video_path)
    tmp = os.path.join("video_data", "112x112")

    if not os.path.exists(tmp):
        os.mkdir(tmp)

    for video in train:
        cap = cv2.VideoCapture(os.path.join(data360.train_video_path, video))
        sample = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, dsize=(112, 112))  # 원래 이미지의 fx, fy배
                sample.append(frame)
            else:
                cap.release()
                break
        name = video.split(".")[0]
        np.save(os.path.join("video_data", "112x112", f"{name}.npy"), np.array(sample))

    for video in test:
        cap = cv2.VideoCapture(os.path.join(data360.test_video_path, video))
        sample = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, dsize=(448, 224))  # 원래 이미지의 fx, fy배
                sample.append(frame)
            else:
                cap.release()
                break
        name = video.split(".")[0]
        np.save(os.path.join("video_data", "448x224", f"{name}.npy"), np.array(sample))


def train_saliency():
    def scheduler(epoch):
        if epoch < 10:
            return 0.001
        else:
            return 0.001 * tf.math.exp(0.1 * (10 - epoch))

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    board_callback = tf.keras.callbacks.TensorBoard(
        log_dir='log/sal', write_graph=True, write_images=False,
        update_freq='batch')

    test_board_callback = tf.keras.callbacks.TensorBoard(
        log_dir='log/random_test', write_graph=True, write_images=False,
        update_freq='batch')

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("weights", "sal"), monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch')

    data360 = Sal360()

    model = make_densenet_model2()
    # model.load_weights(os.path.join("weights", "random0612"))
    # model = tf.keras.models.load_model(os.path.join("weights", "random0612"))
    # model.summary()

    video_path = os.listdir("video_data")
    saliency_path = os.listdir("saliency_data")
    dataset = {}
    saliency_dataset = {}
    for path in video_path:
        name = path.split(".")[0]
        tmp = np.load(os.path.join("video_data", path))
        tmp_saliecny = np.load(os.path.join("saliency_data", "32x32", path))

        # preprocess_tmp = tf.keras.applications.densenet.preprocess_input(tmp)
        dataset[name] = tmp
        saliency_dataset[name] = tmp_saliecny

    # video_data
    train_gen = generator(data=data360, segment=2)
    val_gen = generator(data=data360, mode="val", segment=2)
    test_gen = generator(data=data360, mode="test", segment=2)
    hist = model.fit(train_gen, steps_per_epoch=630 * 9, validation_data=val_gen, validation_steps=12 * 14 * 9,
                     validation_freq=2, epochs=1000,
                     callbacks=[board_callback, ckpt_callback, lr_callback])


def hello2():
    enc, dec = restore_model()
    model = tf.keras.models.load_model(os.path.join("weights", "mae_ed_sal2_train"))
    # model.summary()
    # exit()
    enc.load_weights(os.path.join("weights", "mae_ed_sal2_enc", "weights.h5"))
    dec.load_weights(os.path.join("weights", "mae_ed_sal2_dec", "weights.h5"))
    # enc.summary()
    # dec.summary()
    data360 = Sal360()
    video_path = os.listdir("video_data/224x224")
    dataset = {}
    saliency_dataset = {}

    for path in video_path:
        name = path.split(".")[0]
        tmp = np.load(os.path.join("video_data/224x224", path))
        tmp_saliecny = np.load(os.path.join("saliency_data", "32x32", path))
        # dataset[name] = tf.keras.applications.densenet.preprocess_input(tmp)
        dataset[name] = tmp / 255.
        saliency_dataset[name] = tmp_saliecny
    # model = tf.keras.models.load_model(os.path.join("weights", "ed"))
    # model.summary()
    test_video = os.listdir(data360.test_video_path)
    test_true_data = data360.test[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for video in test_video:
        target_true_data = test_true_data[video]  # 57 100 7
        target_video = dataset[video.split(".")[0]]
        target_saliency = saliency_dataset[video.split(".")[0]]
        for trajectory in target_true_data:  # 100 7
            # enc.reset_states()
            video_data = []
            saliency_data = []
            position_data = []
            for true in trajectory:
                lng, lat, sf, ef = true[1], true[2], int(true[5]), int(true[6])
                video_data.append(target_video[sf])
                saliency_data.append(target_saliency[sf])
                position_data.append([lng, lat])
            degree = list(map(toDegree, position_data))
            video_perspective_view = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in
                                      zip(video_data, degree)]
        #     state = enc([np.array([video_data[:50]]), np.array([video_perspective_view[:50]])])
        #     pred_val = dec([np.array([position_data[:50]])] + state)
        #     pred_val = np.reshape(pred_val, [-1, 2])[:5]
        #     predicted = []
        #     for i in range(10):
        #         pred_val = np.reshape(pred_val, [-1, 2])
        #         degree = list(map(toDegree, pred_val))
        #         x1 = np.array([video_data[50 + (i * 5): 50 + ((i + 1) * 5)]])
        #         x2 = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in
        #               zip(video_data[50 + (i * 5): 50 + ((i + 1) * 5)], degree)]
        #         state = enc([x1, np.array([x2])])
        #         pred_val = dec([np.array([pred_val])] + state)
        #         # pred_val = np.reshape(pred_val, [-1, 2]).tolist()
        #         predicted.append(pred_val)
        #     _true = np.array(position_data)
        #     predicted = np.reshape(predicted, [-1, 2])
        #     # _pred = np.array(predicted)
        #     plt.plot(_true[:, 0], c="b")
        #     plt.plot(list(range(50, 100)), predicted[:, 0], c="r")
        #     plt.show()
        #     plt.plot(_true[:, 1], c="b")
        #     plt.plot(list(range(50, 100)), predicted[:, 1], c="r")
        #     plt.show()
        #
        #     # x2 = np.array([video_perspective_view[50 + i * 5: 50 + (i + 1) * 5]])
        #     # x3 = np.array([position_data[50 + i * 5:50 + (i + 1) * 5]])
        #
        # x1 = np.array([video_data[:5]])
        # x2 = np.array([video_perspective_view[:5]])
        # x3 = np.array([position_data[:5]])
        # true_pos = position_data[5:10]
        #
        # pred_val = np.reshape(pred_val, [-1, 2])
        # plt.plot(true_pos, c='b')
        # plt.plot(pred_val, c="r")
        # plt.show()
        # # pred_pos = model([x1, x2, x3])
            pred_data = []
        # x1 = np.array([video_data[:55]])
        # x2 = np.array([video_perspective_view[:55]])
        # x3 = np.array([position_data[45:50]])
        # state = enc([x1, x2])
        # target_sequence = np.array([position_data[50:55]])
        # __x = np.array(position_data[50:55])
        # pred_val, h, c = dec([target_sequence] + state)
        # pred_val = np.reshape(pred_val, [-1, 2])
        # pred_val = np.clip(pred_val, 0, 1)
        # plt.plot(pred_val[:, 0], c='r')
        # plt.plot(__x[:, 0], c='b')
        # plt.ylim(0, 1)
        # plt.show()
        # plt.plot(pred_val[:, 1], c='r')
        # plt.plot(__x[:, 1], c='b')
        # plt.ylim(0, 1)
        # plt.show()
            x1 = np.array([saliency_data[:50]])
            x2 = np.array([video_perspective_view[:50]])
            x3 = np.array([position_data[:50]])
            dec_in = np.array([position_data[45:50]])

            state = enc([x1, x2, x3])
            pred_val, h, c = dec([dec_in] + state)
            pred_val = np.reshape(pred_val, [-1, 2])
            plt.plot(pred_val[:, 0], c="b")
            plt.plot(position_data[:][0], c="r")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            plt.plot(pred_val[:, 1], c="b")
            plt.plot(position_data[:][1], c="r")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.show()
            # target_sequence = position_data[0]
            #
            # for i in range(100):
            #     x1 = np.array([[video_data[i]]])
            #     x2 = np.array([[video_perspective_view[i]]])
            #     x3 = np.array([[position_data[i]]])
            #     # target_sequence = position_data[:i]
            #     state = enc([x1, x2])
            #     pred_val, _, _ = dec.predict([x3] + state)
            #     # print(pred_val)
            #     # state = [h, c]
            #     pred_val = np.reshape(pred_val, [-1, 2])[0].tolist()
            #     target_sequence += pred_val
            #     pred_data += pred_val
            #     print(pred_val, position_data[i])
            # degree = list(map(toDegree, pred_val[0, :]))
            # video_perspective_view = [py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 64)) for frame, pp in
            #                           zip(video_data[i], degree)]

            # x1 = np.array([video_data[i]])
            # x2 = np.array([video_perspective_view])
            # state = enc([x1, x2])

            # target_sequence = pred_val
            # pred_data.append(pred_val)
            # pred_pos = model([np.array([video_data[1:51]]), pred_pos])
            # pred_pos = model([np.array([video_data[2:52]]), pred_pos])
            # pred_pos = model([np.array([video_data[3:53]]), pred_pos])
            # pred_pos = model([np.array([video_data[4:54]]), pred_pos])
            # pred_pos = model([np.array([video_data[5:55]]), pred_pos])

            # pred_pos = np.reshape(pred_pos, [-1, 2])
            # pred_pos2 = model([np.array([video_data[50:55]]), np.array([pred_pos[-5:]])])
            # pred_pos3 = model([np.array([video_data[55:60]]), pred_pos2])
            # pred_data = np.reshape(pred_data, (-1, 2))
            # print(np.square(position_data - pred_data).mean())
            # pred_pos = pred_data.tolist()
            # pred_pos2 = np.reshape(pred_pos2, [-1, 2]).tolist()
            # pred_pos3 = np.reshape(pred_pos3, [-1, 2]).tolist()

            # pred_pos = pred_pos + pred_pos2 + pred_pos3
            name = video.split(".")[0]
            pred_data = np.reshape(pred_data, [-1, 2])
            position_data = np.array(position_data)
            plt.title(f"{name}")
            plt.plot(list(range(0, 100)), position_data[:, 0], c="b", label="true lng")
            plt.plot(list(range(0, 100)), pred_data[:, 0], c="r", label="pred lng")
            plt.legend(loc="lower right")
            plt.ylim(0, 1)
            plt.show()

            plt.title(f"{name}")
            plt.plot(list(range(0, 100)), position_data[:, 1], c="b", label="true lat")
            plt.plot(list(range(0, 100)), pred_data[:, 1], c="r", label="pred lat")
            plt.legend(loc="lower right")
            plt.ylim(0, 1)
            plt.show()


def toDegree(x):
    return (np.array(x) * 2 - 1) * np.array([180, 90])


def make_densenet_model5():
    input_shape = (1, None, 224, 224, 3)
    input_shape2 = (1, None, 64, 64, 3)

    state_in = Input(batch_shape=input_shape)
    state_in2 = Input(batch_shape=input_shape2)
    # o_in = Input(batch_shape=(1, None, 2))
    # feature = keras.applications.DenseNet121(include_top=False, pooling="avg")
    # x = TimeDistributed(feature)(state_in)
    # x = Flatten()(x)

    x = ConvLSTM2D(64, 3, 1, padding='same', return_sequences=True, stateful=True)(state_in)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, 3, 2, padding='same', return_sequences=True, stateful=True)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling3D()(x)
    x = ConvLSTM2D(32, 3, 1, padding='same', return_sequences=True, stateful=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(32, 3, 2, padding='same', return_sequences=False, stateful=True)(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = BatchNormalization()(x)
    x = Flatten()(x)

    x2 = ConvLSTM2D(16, 3, 1, padding='same', return_sequences=True, stateful=True)(state_in2)
    x2 = BatchNormalization()(x2)
    x2 = ConvLSTM2D(16, 3, 2, padding='same', return_sequences=True, stateful=True)(x2)
    x2 = BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling3D()(x2)
    x2 = ConvLSTM2D(8, 3, 1, padding='same', return_sequences=True, stateful=True)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ConvLSTM2D(8, 3, 1, padding='same', return_sequences=False, stateful=True)(x2)
    x2 = tf.keras.layers.MaxPooling2D()(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)
    concat = keras.layers.concatenate([x, x2])
    concat = tf.keras.layers.RepeatVector(5)(concat)
    # concat = keras.layers.concatenate([concat])
    y = Bidirectional(LSTM(128, return_sequences=True, stateful=True))(concat)
    y = Bidirectional(LSTM(128, return_sequences=True, stateful=True))(y)

    x = TimeDistributed(Dense(100, activation="relu"))(y)
    x = TimeDistributed(Dense(2, activation="linear"))(x)
    # adam = tf.keras.optimizers.Adam(1e-5)
    m = keras.models.Model(inputs=[state_in, state_in2], outputs=x)
    m.compile(optimizer='adam', loss="mse", metrics=['accuracy', 'mae'])
    m.summary()
    return m


if __name__ == '__main__':
    hello2()
    # data360 = Sal360()
    # hello()
    # make_densenet_model5()
    # makesalmap()
    # makedd()
    # g = generator(data360)
    # dataset = {}
    # video_path = os.listdir("video_data/448x224")
    # e = NFOV(64, 64)
    # for path in video_path:
    #     name = path.split(".")[0]
    #     tmp = np.load(os.path.join("video_data", "448x224", path))
    #     if len(tmp) == 601:
    #         tmp = tmp[:-1]
    #     if len(tmp) == 501:
    #         tmp = tmp[:-1]
    #     dataset[name] = tmp
    # for a, b in g:
    #     for frame, pp in zip(a[0][0], a[1][0]):
    #         pp = (np.array(pp) * 2 - 1) * np.array([180, 90])
    #         # ef = GetPerspective(frame, 110, pp[0], pp[1], 64, 64)
    #         # ef = Equirectangular(frame).GetPerspective(110, pp[0], pp[1], 64, 64)
    #         ef = py360convert.e2p(frame, (110, 110), pp[0], pp[1], (64, 128))
    #         cv2.imshow("eframe", ef)
    #         cv2.imshow("frame", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
