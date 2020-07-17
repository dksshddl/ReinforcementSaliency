import os
import time
import random
import re

import threading
import tensorflow as tf
import numpy as np
import cv2
import scipy, multiprocessing
import imageio

from networks.srgan import get_G, get_D

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = 8  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = 1e-4
beta1 = 0.9
## initialize G
n_epoch_init = 100
## adversarial learning (SRGAN)
n_epoch = 2000
lr_decay = 0.1
decay_every = int(n_epoch / 2)
shuffle_buffer_size = 128

## train set location
hr_img_path = 'DIV2K/DIV2K_train_HR/'
lr_img_path = 'DIV2K/DIV2K_train_LR_bicubic/X4/'

## test set location
hr_img_path = 'DIV2K/DIV2K_valid_HR/'
lr_img_path = 'DIV2K/DIV2K_valid_LR_bicubic/X4/'

checkpoint_dir = "weights/srgan"

if not os.path.exists(checkpoint_dir):
    os.mkdir(os.mkdir(checkpoint_dir))


def load_file_list(path, regx, printable=True, keep_prefix=False):
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)
    return return_list


def threading_data(data=None, fn=None, **kwargs):
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    ## start multi-threaded reading.
    results = [None] * len(data)  ## preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
            name='threading_and_return',
            target=apply_fn,
            args=(results, i, data[i], kwargs)
        )
        t.start()
        threads.append(t)

    ## <Milo> wait for all threads to complete
    for t in threads:
        t.join()

    return np.asarray(results)


def read_images(img_list, path='', n_threads=10, printable=True):
    """Returns all images in list by given path and name of each image file.

    Parameters
    -------------
    img_list : list of str
        The image file names.
    path : str
        The image folder path.
    n_threads : int
        The number of threads to read image.
    printable : boolean
        Whether to print information when reading images.

    Returns
    -------
    list of numpy.array
        The images.
    """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx:idx + n_threads]
        b_imgs = threading_data(b_imgs_list, fn=read_images, path=path)
        # tl.logging.info(b_imgs.shape)
        imgs.extend(b_imgs)
    return imgs


def read_image(image, path=''):
    """Read one image.

    Parameters
    -----------
    image : str
        The image file name.
    path : str
        The image folder path.

    Returns
    -------
    numpy.array
        The image.

    """
    return imageio.imread(os.path.join(path, image))


def get_train_data():
    # load dataset

    train_hr_img_list = sorted(load_file_list(path=hr_img_path, regx='.*.png', printable=False))  # [0:20]
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = read_images(train_hr_img_list, path=hr_img_path, n_threads=32)

    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img

    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
    # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds


def train():
    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tf.keras.applications.VGG19(include_top=False)
    model = tf.keras.Sequential()
    for layer in VGG.layers:
        model.add(layer)
        if layer.name == "block4_pool":
            break
    del VGG
    model.summary()

    lr_v = tf.Variable(lr_init)

    g_optimizer_init = tf.train.AdamOptimizer(lr_v, beta1=beta1)
    g_optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)
    d_optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()

    ## initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mse_loss = tf.losses.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        # if (epoch != 0) and (epoch % 10 == 0):
        #     tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4],
        #                        os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))

    ## adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            if lr_patchs.shape[0] != batch_size:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs + 1) / 2.)  # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs + 1) / 2.)
                d_loss1 = tf.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tf.losses.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                g_gan_loss = 1e-3 * tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tf.losses.mean_squared_error(fake_patchs, hr_patchs)
                vgg_loss = 2e-6 * tf.losses.mean_squared_error(feature_fake, feature_real)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            print(
                "Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss,
                    d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = f" ** new learning rate: {lr_init * new_lr_decay} (for GAN)"
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            # tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))
