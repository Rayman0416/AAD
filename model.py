#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_processing.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/15 1:13   lintean      1.0         None
'''

import math
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from dotmap import DotMap
from utils import cart2sph, pol2cart, makePath
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from scipy.io import loadmat
import os
from importlib import reload
import shap
np.set_printoptions(suppress=True)


def get_logger(name, log_path):
    import logging
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, args):
    locs = loadmat('locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def extract_shap(image, args):
    locs = loadmat('locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = np.array([azim_proj(e) for e in locs_3d])

    x_min, x_max = locs_2d[:,0].min(), locs_2d[:,0].max()
    y_min, y_max = locs_2d[:,1].min(), locs_2d[:,1].max()

    x_idx = ((locs_2d[:,0] - x_min) / (x_max - x_min) * (args.image_size - 1)).round().astype(int)
    y_idx = ((locs_2d[:,1] - y_min) / (y_max - y_min) * (args.image_size - 1)).round().astype(int)

    channel_shap = image[x_idx, y_idx]

    return channel_shap

def recreate_images(data, args, top_indices):
    locs = loadmat('locs_orig.mat')
    locs_3d = locs['data'][top_indices, :]
    locs_2d = np.array([azim_proj(e) for e in locs_3d])
    data = data[:, top_indices]

    
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)

    return images


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2 * args.audio_channel:]

            data.append(eeg_data)

    data = pd.concat(data, axis=0, ignore_index=True)
    return data


# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split(data, args):
    random.seed(args.random_seed)
    # init
    test_percent = args.test_percent
    window_lap = args.window_length * (1 - args.overlap)
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)

    train_set = []
    test_set = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")

        # split trial
        for k in range(args.trail_number):
            # the number of windows in a trial
            window_number = math.floor(
                (args.cell_number - args.window_length) / window_lap) + 1

            test_window_length = math.floor(
                (args.cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1

            test_window_left = random.randint(0, window_number - test_window_length)
            test_window_right = test_window_left + test_window_length - 1
            target = label.iloc[k, args.label_col]

            # split window
            for i in range(window_number):
                left = math.floor(k * args.cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # train set or test set
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, args.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, args.subject_number]))

    # concat
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set


def to_alpha(data, window, args):
    alpha_data = []
    for window_index in range(window.shape[0]):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        window_data = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        window_data = np.abs(window_data) / args.window_length
        window_data = np.sum(np.power(window_data[args.point_low:args.point_high, :], 2), axis=0)
        alpha_data.append(window_data)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data


def main(name="S1", data_document_path="../KUL_single_single3"):
    args = DotMap()
    args.reduce = 32
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * 1)
    args.overlap = 0.8
    args.batch_size = 32
    args.max_epoch = 100
    args.random_seed = time.time()
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "./result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    logger = get_logger(args.name, args.log_path)

    # load data 和 label
    data = read_prepared_data(args)

    # split window、testset
    data, train_window, test_window = window_split(data, args)
    train_label = train_window[:, args.window_metadata.target]
    test_label = test_window[:, args.window_metadata.target]

    # fft
    train_data = to_alpha(data, train_window, args)
    test_data = to_alpha(data, test_window, args)
    train_alpha_data = train_data
    test_alpha_data = test_data
    del data

    # to images
    train_data = gen_images(train_data, args)
    test_data = gen_images(test_data, args)

    # add 1 channel(grayscale) dimension for conv2d keras: (batch_size, height, width, channels)
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    # encode labels like [1, 0] and [0, 1] for softmax
    train_label = to_categorical(train_label - 1, 2)
    test_label = to_categorical(test_label - 1, 2)

    # train
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_data.shape[1:],
                     kernel_regularizer=regularizers.l2(0.01), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_last"))
    model.add(Dropout(0.1))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output layer
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # Output the parameter status of each layer of the model
    model.summary()

    opt = RMSprop(lr=0.0003, decay=3e-4)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    # plot_model(model, to_file='model.png', show_shapes=True)

    results = {}
    history = model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.max_epoch, validation_split=args.vali_percent, verbose=2)
    loss, accuracy = model.evaluate(test_data, test_label)
    print(loss, accuracy)

    results[args.eeg_channel] = {
        'loss': loss,
        'accuracy': accuracy,
    }

    # Shapley value analysis
    background = train_data[np.random.choice(train_data.shape[0], 200, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    to_explain = test_data[:100]
    shap_values = explainer.shap_values(to_explain)
    true_classes = np.argmax(test_label[:100], axis=1)
    
    # Collect Shapley values for the correct classes
    correct_class_shap = []
    for i, cls in enumerate(true_classes):
        correct_class_shap.append(shap_values[cls][i])
    
    mean_shap = np.mean(correct_class_shap, axis=0)
    mean_shap = np.squeeze(mean_shap, axis=-1)
    mean_shap = extract_shap(mean_shap, args)

    sorted_indices = np.argsort(mean_shap)[::-1]

    reduction_list = [32, 16]
    for reduction in reduction_list:
        top_indices = sorted_indices[:reduction]
        print(f"Top {reduction} indices:", top_indices)
        train_data = recreate_images(train_alpha_data, args, top_indices)
        test_data = recreate_images(test_alpha_data, args, top_indices)

        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)
        print("Recreated images shape:", train_data.shape)

        # Reduce features based on Shapley values
        model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.max_epoch, validation_split=args.vali_percent, verbose=2)
        loss_reduced, accuracy_reduced = model.evaluate(test_data, test_label)

        results[reduction] = {
            'loss': loss_reduced,
            'accuracy': accuracy_reduced
        }

    return results


if __name__ == "__main__":
    main()

