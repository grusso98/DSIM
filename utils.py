import os
import re
from random import random

import tensorflow as tf
import matplotlib as plt
import keras
import numpy as np
import PIL
from scipy.signal import decimate
from scipy import interpolate
import librosa
import wave
import soundfile as sf
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.layers import Lambda, Conv1D, Lambda
from keras.layers import LeakyReLU
import tensorflow as tf


def scale(img):
    if keras.backend.max(img) > 1:
        return img / 255.0
    else:
        return img


def rgb_to_ycbcr(img):
    img_ycbcr = tf.image.rgb_to_yuv(img)[:, :, :, 0]
    return tf.expand_dims(img_ycbcr, axis=3)


def downscale(img, ds_factor):
    return tf.image.resize(img, [img.shape[1] // ds_factor, img.shape[2] // ds_factor], method="area")


def gt_lr_tuple(img_hr, ds_factor):
    print(img_hr.shape)
    img_lr = downscale(img_hr, ds_factor)
    print(img_lr.shape)
    return (img_lr, img_hr)


def preprocessing(dataset, channel="rgb", ds_factor=4, scale=True):
    if scale == True:
        dataset = dataset.map(lambda x: scale(x))

    if channel == "ycbcr":
        dataset = dataset.map(lambda x: rgb_to_ycbcr(x))

def get_lowres_image(img, upscale_factor=4):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


def upscale_image(model, img, channels="rgb", fix='None'):
    """Predict the result based on input image and restore the image as RGB."""
    up_factor = 4
    if channels == "rgb":
        y = tf.keras.preprocessing.image.img_to_array(img)
        #y = y.astype("float32") / 255.0
        input = np.expand_dims(y, axis=0)

        out = model.predict(input)

        out_img_y = out[0]

        if not np.max(out_img_y) > 10:
            out_img_y *= 255.0

        out_img_y = out_img_y.clip(0, 255)
        out_img = PIL.Image.fromarray(np.uint8(out_img_y))

    if channels == "ycbcr":
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = tf.keras.preprocessing.image.img_to_array(y)
        #y = y.astype("float32") / 255.0

        input = np.expand_dims(y, axis=0)
        out = model.predict(input)

        out_img_y = out[0]
        if not np.max(out_img_y) > 10:
            out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
            "RGB")
    if fix == 'RGB correction':
        pass
    if fix == 'YCbCr correction':
        pass
    return out_img

def upsample(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f)

    return x_sp

def preprocess(file_list, start, end, sr=48000, scale=6, dimension=256, stride=256, tag='test'):
    #random.shuffle(file_list)
    data_size = end - start + 1
    lr_patches = list()
    hr_patches = list()
    dataset_name = None
    for i, wav_path in enumerate(file_list[start:end + 1]):
        if i % 10 == 0: print("%s - %d/%d" % (wav_path, i + 1 + start, len(file_list)))

        # Get low sample rate version data for training
        x_hr, fs = librosa.load(wav_path, sr=sr)
        x_len = len(x_hr)
        x_hr = x_hr[: x_len - (x_len % scale)]

        # Down sampling for Low res version
        x_lr = decimate(x_hr, scale)
        # x_lr = np.array(x_hr[0::scale])

        # Upscale using cubic spline Interpolation
        x_lr = upsample(x_lr, scale)

        x_lr = np.reshape(x_lr, (len(x_lr), 1))
        x_hr = np.reshape(x_hr, (len(x_hr), 1))

        for i in range(0, x_lr.shape[0] - dimension, stride):
            lr_patch = x_lr[i:i + dimension]

            # mid = dimension // 2 - stride // 2
            # hr_patch = x_hr[i+mid:i+mid+stride]

            hr_patch = x_hr[i:i + dimension]

            lr_patches.append(lr_patch)
            hr_patches.append(hr_patch)

    hr_len = len(hr_patches)
    lr_len = len(lr_patches)

    hr_patches = np.array(hr_patches[0:hr_len])
    lr_patches = np.array(lr_patches[0:lr_len])

    print('high resolution(Y) dataset shape is ', hr_patches.shape)
    print('low resolution(X) dataset shape is ', lr_patches.shape)

    dataset_name = 'drive/MyDrive/DSIM_project/Audio-SuperRes/audio_samples/monospeaker/asr-ex%d-start%d-end%d-scale%d-sr%d-dim%d-strd%d-%s.h5' % (
        data_size,
        start,
        end,
        scale,
        sr,
        dimension,
        stride,
        tag
    )

    return lr_patches, hr_patches, dataset_name

def load_model(model, weights_file, load_weights=False):
    if load_weights:
        print(weights_file)
        model.load_weights(weights_file)
        print('load model weights success!')
    return model

def SNR(y_true, y_pred):
    P = y_pred
    Y = y_true
    sqrt_l2_loss = K.sqrt(K.mean((P - Y) ** 2 + 1e-6))
    sqrn_l2_norm = K.sqrt(K.mean(Y ** 2))
    snr = 20 * K.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / K.log(10.)
    avg_snr = K.mean(snr)
    return avg_snr

def sum_loss(y_true, y_pred):
    P = y_pred
    Y = y_true
    loss = K.sum((P - Y) ** 2)
    return loss

def compile_model(model):
    model.compile(loss='mse', optimizer="adam", metrics=[sum_loss, SNR])
    return model

def load_wav_list(dirname):
    file_list = []
    filenames = os.listdir(dirname)
    file_extensions = set(['.flac'])
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext in file_extensions:
            full_filename = os.path.join(dirname, filename)
            file_list.append(full_filename)


    print('load flac list examples..')

    for i, file in enumerate(file_list):
        print(file)

        if i > 5: break

    return file_list

def audio(audio_path, type):
    BATCH_SIZE = 256
    LOAD_WEIGHTS = True
    WEIGHTS_PATH = 'models/audio_models/'

    last_slash = [m.start() for m in re.finditer('/', audio_path)]
    audio_path = audio_path[0:last_slash[-1]]
    print('path: ' + audio_path)
    if type == "single":
        WEIGHTS_FILE = 'asr-weights-k32.hdf5'
        model = base_model(summary=False)
        model = load_model(model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), load_weights=LOAD_WEIGHTS)
        model = compile_model(model)
    if type == "multi":
        WEIGHTS_FILE = 'asr-weights-4multi-stride256.hdf5'
        model = base_model(summary=False)
        model = load_model(model, os.path.join(WEIGHTS_PATH, WEIGHTS_FILE), load_weights=LOAD_WEIGHTS)
        model = compile_model(model)

    # load test wav samples
    test_samples = load_wav_list(audio_path)

    # patch sample data
    X, Y, _ = preprocess(test_samples, start=0, end=len(test_samples) - 1, sr=48000, scale=4, dimension=256,
                         stride=256, tag='test')

    print(X.shape)
    print(Y.shape)

    # predict
    pred = model.predict(X)
    # evaluate
    scores = model.evaluate(X, Y)
    print(scores)
    print('Evaluate scores')
    for score in scores:
        print('- %10f' % (score))

    snr_spline = tf.image.psnr(Y, X, 1)
    snr_sr = tf.image.psnr(Y, pred, 1)
    print(snr_spline)
    print(snr_sr)
    audio_path = audio_path + type
    sf.write(audio_path + 'original.flac', Y.flatten(), 48000, 'PCM_24')
    sf.write(audio_path + 'downsampled.flac', X.flatten(), 48000, 'PCM_24')
    sf.write(audio_path +'superrezzed.flac', pred.flatten(), 48000, 'PCM_24')

def split(x):
    return x[:, 28:36]  # it is fixed range for input(64) & output(8) dataset

def SubPixel1D(input_shape, r, color=False):
    def _phase_shift(I, r=2):
        X = tf.transpose(I, [2, 1, 0])  # (r, w, b)
        X = tf.batch_to_space(X, [r], [[0, 0]])  # (1, r*w, b)
        X = tf.transpose(X, [2, 1, 0])
        return X

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * r,
                int(input_shape[2] / (r))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        # only single channel!
        x_upsampled = _phase_shift(x, r)
        return x_upsampled

    return Lambda(subpixel, output_shape=subpixel_shape)

def base_model(summary=True):
    print('load base model..')
    x = keras.layers.Input((256, 1))
    main_input = x

    # 128 256 512 512
    # 65 31 15 15

    # Donwsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=16, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x1 = x  # 128

    # Donwsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x2 = x  # 64

    # Donwsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x3 = x  # 32

    # Donwsampling layer 4
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x4 = x  # 16

    # Donwsampling layer 5
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x5 = x  # 8

    # Donwsampling layer 6
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x6 = x  # 4

    # Bottleneck layer
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32, kernel_size=32, activation=None,
               strides=2)(x)
    x = LeakyReLU(0.2)(x)

    # Upsampling layer 6
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 32, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x6])

    # Upsampling layer 5
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 32, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x5])

    # Upsampling layer 4
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 32, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x4])

    # Upsampling layer 3
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 32, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x3])

    # Upsampling layer 2
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 32, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x2])

    # Upsampling layer 1
    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=2 * 16, kernel_size=32, activation=None)(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.5)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    x = keras.layers.concatenate([x, x1])

    # SubPixel-1D Final
    x = Conv1D(padding='same', kernel_initializer='he_normal', filters=2, kernel_size=32, activation=None)(x)
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    output = keras.layers.add([x, main_input])
    model = keras.models.Model(main_input, output)

    if summary:
        model.summary()

    return model

def denorm_n11_01(x):
    return tf.math.scalar_mul(1/2,tf.math.add(x,1))

def denorm_n11_255(x):
    return tf.math.scalar_mul(127.5,tf.math.add(x,1))

def norm_n11(x):
    return tf.math.subtract(tf.math.scalar_mul(1/127.5,x),1)

def norm_01(x):
    return tf.math.scalar_mul(1/255.0,x)

def denorm_01_255(x):
    return tf.math.scalar_mul(255.0,x)