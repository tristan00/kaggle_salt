import pandas as pd
import glob
from PIL import Image
import numpy as np
import random
import functools
import operator
import keras.utils
import scipy.misc
import traceback
import multiprocessing
from scipy.ndimage import gaussian_gradient_magnitude, morphology
from scipy.ndimage.morphology import binary_opening
from keras import optimizers
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.layers import Input
import math
from sklearn.svm import SVC
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
import os
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import h5py
import configparser
from sklearn import preprocessing
import ast
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler
import shutil
import gc
from scipy import ndimage
from sklearn.model_selection import train_test_split
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.feature import hog
from skimage import exposure

train_files_loc = '/home/td/Documents/salt/train/'
test_files_loc = '/home/td/Documents/salt/test/'
depth_loc = '/home/td/Documents/salt/depths.csv'

full_image_read_size = (101, 101)
border = 5

def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2 * (intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


def ConvBlock(model, layers, filters):
    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))



def get_segnet():
    pass

def get_vgg():
    input_img = Input((128, 128, 5), name='img')
    input_features = Input((1,), name='feat')

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    # c1 = Dropout(0.5)(c1)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    # c2 = Dropout(0.5)(c2)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    # c3 = Dropout(0.5)(c3)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    # c4 = Dropout(0.5)(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Join features information in the depthest layer
    f_repeat = RepeatVector(8 * 8)(input_features)
    f_conv = Reshape((8, 8, 1))(f_repeat)
    p4_feat = concatenate([p4, f_conv], -1)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4_feat)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    # c6 = Dropout(0.2)(c6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    # c7 = Dropout(0.2)(c7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    # c8 = Dropout(0.2)(c8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    # c9 = Dropout(0.2)(c9)

    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(input=[input_img, input_features], output=[outputs])
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc', IOU_calc])
    return model


def gaussian_noise(np_image, num):
    if num == 0:
        return np_image
    else:
        rand_array = np.random.normal(0,1,101*101)
        rand_array = np.reshape(rand_array, (101, 101))
        return np.add(np_image, rand_array)


def rescale_img(np_image):
    flat_image = np_image.flatten()
    flat_image = np.reshape(flat_image, (-1, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 255))
    flat_image = scaler2.fit_transform(flat_image)
    np_image_2 = np.reshape(flat_image, (np_image.shape[0], np_image.shape[1]))
    return np_image_2


def generate_input_image_and_masks(mask_locs, batch_size = 512, max_loop = 2, train = True):
    # folders = list(glob.glob(files_loc + 'stage1_train/*/')) + \
    #           list(glob.glob(files_loc + 'extra_data_processed/*/')) + \
    #           list(glob.glob(files_loc + 'extra_data2_processed/*/'))
    

    random.shuffle(mask_locs)
    df = pd.read_csv(depth_loc)
    x, d, y = [], [], []

    while True:
        for mask_loc in mask_locs:
            if train:
                transpose_l = [0, 1]
                rotation_l = [0, 1, 2, 3]
                noise_l = [0,1, 2, 3]
                rescale_l = [True, False]
            else:
                transpose_l = [0, 1]
                rotation_l = [0, 1, 2]
                noise_l = [0]
                rescale_l = [False]

            for transpose in transpose_l:
                for rotation in rotation_l:
                    for noise in noise_l:
                        for rescale in rescale_l:
                            if len(x) > batch_size:
                                x, d, y = [], [] , []

                            mask_name = os.path.basename(mask_loc).split('.')[0]
                            image_loc = train_files_loc + '/images/{0}.png'.format(mask_name)

                            start_image = Image.open(image_loc).convert('LA')
                            np_image = np.array(start_image.getdata())[:, 0]
                            np_image = np_image.reshape(start_image.size[1], start_image.size[0])
                            if rescale:
                                np_image = rescale_img(np_image)
                            np_image = gaussian_noise(np_image, noise)

                            x_center_mean = np_image[border:-border, border:-border].mean()
                            x_csum = (np.float32(np_image) - x_center_mean).cumsum(axis=0)
                            x_csum -= x_csum[border:-border, border:-border].mean()
                            x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

                            y_center_mean = np_image[border:-border, border:-border].mean()
                            y_csum = (np.float32(np_image) - y_center_mean).cumsum(axis=1)
                            y_csum -= y_csum[border:-border, border:-border].mean()
                            y_csum /= max(1e-3, y_csum[border:-border, border:-border].std())

                            img_eq = exposure.equalize_hist(np_image)
                            np_image = np_image.astype(np.float32)

                            # p2, p98 = np.percentile(np_image, (5, 95))
                            # img_rescale = exposure.rescale_intensity(np_image, in_range=(p2, p98))

                            np_image /= 255.0
                            # np_image = gaussian_noise(np_image, noise)

                            start_mask = Image.open(mask_loc).convert('LA')
                            np_mask = np.array(start_mask.getdata())[:, 0]
                            np_mask = np_mask.reshape(start_mask.size[1], start_mask.size[0])
                            # if np_mask.max() == 0:
                            #     continue
                            np_mask= np_mask // 255


                            if transpose == 1:
                                np_image = np.transpose(np_image)
                                np_mask = np.transpose(np_mask)
                                x_csum =  np.transpose(x_csum)
                                img_eq =  np.transpose(img_eq)
                                y_csum =  np.transpose(y_csum)
                                # img_rescale = np.transpose(img_rescale)

                            np_image = np.rot90(np_image, rotation)
                            np_mask = np.rot90(np_mask, rotation)
                            x_csum = np.rot90(x_csum, rotation)
                            img_eq = np.rot90(img_eq, rotation)
                            y_csum = np.rot90(y_csum, rotation)

                            # img_rescale = np.rot90(img_rescale, rotation)

                            np_image = np.pad(np_image, ((0, 27), (0, 27)), 'constant')
                            np_mask = np.pad(np_mask, ((0, 27), (0, 27)), 'constant')
                            x_csum = np.pad(x_csum, ((0, 27), (0, 27)), 'constant')
                            img_eq = np.pad(img_eq, ((0, 27), (0, 27)), 'constant')
                            y_csum = np.pad(y_csum, ((0, 27), (0, 27)), 'constant')

                            # img_rescale = np.pad(img_rescale, ((0, 27), (0, 27)), 'constant')

                            g = gaussian_gradient_magnitude(np_image, sigma = .4)

                            # np_image[127, 127] = df[df['id'] == mask_name]['z'].values[0]

                            np_image = np.dstack((np.expand_dims(np_image, axis=2),
                                                  np.expand_dims(g, axis=2),
                                                  np.expand_dims(x_csum, axis=2),
                                                  np.expand_dims(img_eq, axis=2),
                                                  np.expand_dims(y_csum, axis=2)
                                                  # np.expand_dims(img_rescale, axis=2)
                                                  ))

                            np_mask = np.expand_dims(np_mask, 2)

                            depth_scaled = df[df['id'] == mask_name]['z'].values[0]/1000

                            x.append(np_image)
                            y.append(np_mask)
                            d.append(depth_scaled)

                            if len(x) == batch_size:

                                yield  {'img': np.array(x), 'feat': np.array(d)}, np.array(y)


def image_gen_test():
    image_locs = list(glob.glob(test_files_loc + '/images/*.png'))
    df = pd.read_csv(depth_loc)

    for image_loc in image_locs:
        mask_name = os.path.basename(image_loc).split('.')[0]

        start_image = Image.open(image_loc).convert('LA')
        np_image = np.array(start_image.getdata())[:, 0]
        np_image = np_image.reshape(start_image.size[1], start_image.size[0])
        x_center_mean = np_image[border:-border, border:-border].mean()
        x_csum = (np.float32(np_image) - x_center_mean).cumsum(axis=0)
        x_csum -= x_csum[border:-border, border:-border].mean()
        x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())
        y_center_mean = np_image[border:-border, border:-border].mean()
        y_csum = (np.float32(np_image) - y_center_mean).cumsum(axis=1)
        y_csum -= y_csum[border:-border, border:-border].mean()
        y_csum /= max(1e-3, y_csum[border:-border, border:-border].std())

        img_eq = exposure.equalize_hist(np_image)
        np_image = np_image.astype(np.float32)
        np_image /= 255.0

        np_image = np.pad(np_image, ((0, 27), (0, 27)), 'constant')
        x_csum = np.pad(x_csum, ((0, 27), (0, 27)), 'constant')
        img_eq = np.pad(img_eq, ((0, 27), (0, 27)), 'constant')
        y_csum = np.pad(y_csum, ((0, 27), (0, 27)), 'constant')

        g = gaussian_gradient_magnitude(np_image, sigma=.4)


        # np_image = np.dstack((np.expand_dims(np_image, axis=2),
        #                       np.expand_dims(g, axis=2)))

        np_image = np.dstack((np.expand_dims(np_image, axis=2),
                              np.expand_dims(g, axis=2),
                              np.expand_dims(x_csum, axis=2),
                              np.expand_dims(img_eq, axis=2),
                              np.expand_dims(y_csum, axis=2)
                              ))

        depth_scaled = df[df['id'] == mask_name]['z'].values[0] / 1000

        yield np_image, mask_name, depth_scaled


def rle_encoding(x):
    x = x[:101,:101]
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def image_to_str(np_output, threshold = .5):
    on_label = False
    np_output = np_output[:100,:100]
    np_output = np_output.flatten()

    print(np_output.shape)

    image_list = []
    temp_image_list = []
    for count, i in enumerate(np_output):
        if i > threshold and not on_label:
            temp_image_list = []
            temp_image_list.append(count)
            on_label = True
        elif i > threshold and on_label:
            temp_image_list.append(count)
        elif i >= threshold and on_label:
            on_label = False
            image_list.append(temp_image_list)

    res_str = ''
    for i in image_list:
        res_str += str(int(i[0]) + 1)
        res_str += ' '
        res_str += str(len(i))
        res_str += ' '
    res_str = res_str[:-1]

    return res_str


def generate_output(outputs, names, threshold = .5):
    output_dicts = list()
    conv = lambda l: ' '.join(map(str, l))  # list -> string

    for count, (i, j) in enumerate(zip(names, outputs)):
        o = j >= threshold
        z = j < threshold
        j[o] = 1
        j[z] = 0

        # print('\n{}'.format(conv(rle_encoding(j))))


        output_dict = {'id':i, 'rle_mask':conv(rle_encoding(j))}
        output_dicts.append(output_dict)
        # print(count)

    return pd.DataFrame.from_dict(output_dicts)


def main():
    print(glob.glob(train_files_loc + '/masks/*.png'))
    train, val = train_test_split(list(glob.glob(train_files_loc + '/masks/*.png')), test_size=.01)
    #
    gen_train = generate_input_image_and_masks(train)
    gen_val = generate_input_image_and_masks(val, train=False)
    #
    #
    #
    # x_train, y_train, d_train = [], [], []
    # for i in gen_train:
    #     temp_x, temp_y, d = i
    #     x_train.append(temp_x)
    #     y_train.append(temp_y)
    #     d_train.append(d)
    #
    #     print(len(x_train))
    #
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # d_train = np.array(d_train)
    #
    #
    # x_val, y_val, d_val = [], [], []
    # for i in gen_val:
    #     temp_x, temp_y, d = i
    #     x_val.append(temp_x)
    #     y_val.append(temp_y)
    #     d_val.append(d)
    #
    #     print(len(x_val))
    #
    # x_val = np.array(x_val)
    # y_val = np.array(y_val)
    # d_val = np.array(d_val)
    # #

    model = get_vgg()
    cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=25,
                                  verbose=0, mode='auto')
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=0, verbose=1, min_lr=.000001)
    mcp_save = keras.callbacks.ModelCheckpoint('model-tgs-salt-1.h5', save_best_only=True, monitor='val_loss', verbose=1)
    model.fit_generator(gen_train, validation_data = gen_val, steps_per_epoch=1000, epochs=1000,
                        callbacks=[cb, reduce_lr_loss, mcp_save],
                        validation_steps=100)

    # reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=0, verbose=1, min_lr=.000001)
    # mcp_save = keras.callbacks.ModelCheckpoint('model-tgs-salt-1.h5', save_best_only=True, monitor='val_loss', verbose=1)
    # model.fit({'img': x_train, 'feat': d_train}, y_train, batch_size=64, epochs=200,
    #           callbacks=[cb, reduce_lr_loss, mcp_save],
    #           validation_data=({'img': x_val, 'feat': d_val}, y_val))
    model.load_weights('model-tgs-salt-1.h5')

    # model.fit(x_train, y_train, validation_data=(x_val, y_val),
    #                     callbacks=[cb], epochs=100, batch_size=512)

    gen_test = image_gen_test()
    x_test, test_names, d_test = [], [], []
    for i in gen_test:
        temp_x, temp_name, d = i
        x_test.append(temp_x)
        test_names.append(temp_name)
        d_test.append(d)

        # print(len(x_test))

    x_test = np.array(x_test)
    d_test = np.array(d_test)

    output = model.predict({'img': x_test, 'feat': d_test})

    out = generate_output(output, test_names, threshold = .5)
    out.to_csv('out2.csv', index = False)




def get_mean():
    a = list(glob.glob(train_files_loc + '/masks/*.png'))

    values = []

    for i in a:
        start_mask = Image.open(i).convert('LA')
        np_mask = np.array(start_mask.getdata())[:, 0]
        np_mask = np_mask.reshape(start_mask.size[1], start_mask.size[0])
        if np_mask.max() == 0:
            continue
        values.append(np_mask.mean())

        np_mask = np_mask/255
        conv = lambda l: ' '.join(map(str, l))  # list -> string
        print('\n{}'.format(conv(rle_encoding(np_mask))))

    print(sum(values)/len(values))



if __name__ == '__main__':
    main()