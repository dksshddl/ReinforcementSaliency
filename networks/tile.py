import os

import cv2
import math
import tensorflow as tf
from tensorflow import keras
from dataset import Sal360
import numpy as np


class MyModel:
    def __init__(self):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
                print(e)

        input_shape = (10, 160, 320, 3)
        self.model = keras.models.Sequential()
        # tf.keras.applications.ResNet50(include_top=False)
        #
        # state_in = tf.keras.layers.Input(batch_shape=[None] + [None] + list(self.state_dim))
        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)  # 1280
        # x = tf.keras.layers.TimeDistributed(feature)(state_in)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        # lstm = tf.keras.layers.LSTM(256)(x)
        # out = tf.keras.layers.Dense(2, activation="sigmoid")(lstm)

        # model.add(keras.layers.Input(batch_shape=(1, 10, 160, 320, 3)))
        # model.add(keras.layers.Embedding(input_dim=100, output_dim=10*160*320*3, mask_zero=True))
        # model.add(keras.layers.TimeDistributed(keras.layers.Flatten(), input_shape=input_shape))
        # model.add(keras.layers.TimeDistributed(keras.layers.Masking(mask_value=-1.)))
        # model.add(keras.layers.TimeDistributed(keras.layers.Reshape(input_shape[1:])))
        # model.add(keras.layers.TimeDistributed(keras.layers.Masking(mask_value=-1., )))
        # model.add(keras.layers.Reshape(target_shape=(-1, 10, 160, 320, 3)))
        self.model.add(keras.layers.ConvLSTM2D(40, 3, padding="same", return_sequences=True, stateful=True, batch_size=1,
                                          input_shape=input_shape))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ConvLSTM2D(40, 3, padding="same", return_sequences=True, stateful=True))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ConvLSTM2D(40, 3, padding="same", return_sequences=True, stateful=True))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ConvLSTM2D(40, 3, padding="same", return_sequences=False, stateful=True))
        self.model.add(keras.layers.BatchNormalization())
        # self.model.add(keras.layers.ConvLSTM2D(32, 3, padding="same", return_sequences=True, stateful=True))
        # self.model.add(keras.layers.BatchNormalization())

        # model.add(tf.keras.layers.TimeDistributed(keras.layers.MaxPool2D((3, 3), 2)))
        # self.model.add(keras.layers.ConvLSTM2D(32, 3, padding="same", return_sequences=False, stateful=True))
        # self.model.add(keras.layers.BatchNormalization())

        self.model.add(keras.layers.Conv2D(1, 3, activation="relu", padding="same"))
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.model.compile(loss="mean_squared_error", optimizer=opt)
        # model.build(input_shape=(None, 10, 160, 320, 3))
        self.model.summary()
        self.data = Sal360()
        # self.data.select_trajectory(fx=1, fy=1, saliency=True)
        self.train, self.test = self.data.get_whole_data()

    def reset_state(self):
        self.model.reset_states()

    def load_weights(self, path=None):
        if path is None:
            path = os.path.join("weights", "tile", "tile_350.ckpt")
        self.model.load_weights(path)

    def train_model(self, steps=None):
        x_train, y_train = self.train[0], self.train[1]
        x_test, y_test = self.test[0], self.test[1]

        # tensor_boarder = keras.callbacks.TensorBoard(log_dir=os.path.join("log", "tile"), write_graph=True,
        #                                              batch_size=1,
        #                                              update_freq="epoch")
        # tensor_boarder_result = keras.callbacks.TensorBoard(log_dir=os.path.join("log", "tile"), write_graph=True,
        #                                                     batch_size=1,
        #                                                     update_freq="epoch")
        if not os.path.exists(os.path.join("weights", "tile")):
            os.mkdir(os.path.join("weights", "tile"))
        if not os.path.exists(os.path.join("log", "tile")):
            os.mkdir(os.path.join("log", "tile"))
        writer = tf.compat.v1.summary.FileWriter(os.path.join("log", "tile"))
        writer.add_graph(tf.get_default_graph())
        # x_train = [np.reshape(x, [-1, 10, 160, 320, 3]) for x in x_train]
        # y_train = [np.reshape(y, [-1, 10, 160, 320, 1]) for y in y_train]

        if steps is None:
            steps = 1000
        i = 0
        for epoch in range(steps):
            epoch_loss = []
            for _x, _y in zip(x_train, y_train):
                loss = self.model.fit_generator(DataGenerator(_x, _y, 10), shuffle=False)
                epoch_loss.append(loss.history["loss"][0])
                reward_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss.history["loss"][0])])
                writer.add_summary(reward_summary, i)
                i += 1
                self.reset_state()
            loss_summary = tf.Summary(value=[tf.Summary.Value(tag="epoch_loss", simple_value=np.mean(epoch_loss))])
            writer.add_summary(loss_summary, epoch)

            if i != 0 and i % 25 == 0:
                self.model.save_weights(os.path.join("weights", "tile", "tile_" + str(i) + ".ckpt"))

        # x_test = [np.reshape(x, [-1, 10, 160, 320, 3]) for x in x_test]
        # y_test = [np.reshape(y, [-1, 10, 160, 320, 1]) for y in y_test]
        print("test data")
        print(np.shape(x_test), np.shape(y_test))
        for i, (_x, _y) in enumerate(zip(x_test, y_test)):
            pred_callback = PredictCallback()
            pred = self.model.predict_generator(DataGenerator(_x, _y, 10), callbacks=[pred_callback])
            preds = np.reshape(pred_callback.predict_map, [-1, 160, 320, 1])
            _y = np.reshape(_y, [-1, 160, 320, 1])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if not os.path.exists(os.path.join("saliency")):
                os.mkdir(os.path.join("saliency"))
            writer_true = cv2.VideoWriter(os.path.join("saliency", "true_" + str(i) + ".avi"),
                                          fourcc, int(len(_x) // 2), (320, 160), 0)
            writer_pred = cv2.VideoWriter(os.path.join("saliency", "pred_" + str(i) + ".avi"),
                                          fourcc, int(len(_x) // 2), (320, 160), 0)
            for true, pred in zip(_y, preds):
                writer_pred.write(pred)
                writer_true.write(true)
                cv2.imshow("test", true)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.waitKey(27)
            writer_true.release()
            writer_pred.release()
            cv2.destroyAllWindows()

    def eval(self):
        self.load_weights()
        x_test, y_test = self.test[0], self.test[1]

        print("test data")
        print(np.shape(x_test), np.shape(y_test))
        for i, (_x, _y) in enumerate(zip(x_test, y_test)):
            print(np.shape(_y), np.shape(_y))
            pred_callback = PredictCallback()
            pred = self.model.predict_generator(DataGenerator(_x, _y, 10), callbacks=[pred_callback])
            preds = np.reshape(pred_callback.predict_map, [-1, 160, 320, 1])
            _y = np.reshape(_y, [-1, 160, 320, 1])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if not os.path.exists(os.path.join("saliency")):
                os.mkdir(os.path.join("saliency"))
            writer_true = cv2.VideoWriter(os.path.join("saliency", "true_" + str(i) + ".avi"),
                                          fourcc, int(len(_x) // 2), (320, 160), 0)
            writer_pred = cv2.VideoWriter(os.path.join("saliency", "pred_" + str(i) + ".avi"),
                                          fourcc, int(len(_x) // 2), (320, 160), 0)
            for true, pred in zip(_y, preds):
                writer_pred.write(pred)
                writer_true.write(true)
                cv2.imshow("test", true)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                cv2.waitKey(27)
            writer_true.release()
            writer_pred.release()
            cv2.destroyAllWindows()


class PredictCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.predict_map = []

    def on_predict_batch_end(self, batch, logs=None):
        self.predict_map.append(logs['outputs'])

    def on_predict_end(self, logs=None):
        print(logs)
        # output = logs.get("outputs")
        # print(np.shape(output))
        # self.predict_map.append(output)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size  # window size
        self.window_size = batch_size

    def __len__(self):
        return int(len(self.x) - self.batch_size)

    def __getitem__(self, item):
        x = self.x[item:self.window_size]
        y = self.y[self.window_size]
        self.window_size += 1
        return np.array([x]), np.array([y])


if __name__ == '__main__':
    a = MyModel()
    # a.train_model()
    a.eval()
