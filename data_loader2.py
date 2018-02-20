from __future__ import print_function
import tensorflow as tf
from ops import *
import numpy as np


def pre_emph(x, coeff=0.95):
    x0 = tf.reshape(x[0], [1,])
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat(0, [x0, diff])
    return concat

def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

def read_and_decode(filename_queue, canvas_size, preemph=0.):
    reader = tf.TFRecordReader()
    count = 0
    while count < 100:
        try:
            _, serialized_example = reader.read(filename_queue)
            break
        except:
            print('Skipped due to an error in reading {}'.format(filename_queue))
            count = count + 1
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
                'ref_raw': tf.FixedLenFeature([], tf.string),
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(canvas_size)
    wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    noisy.set_shape(canvas_size)
    noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.
    ref = tf.decode_raw(features['ref_raw'], tf.int32)
    ref.set_shape(canvas_size)
    ref = (2./65535.) * tf.cast((ref - 32767), tf.float32) + 1.
    if preemph > 0:
        wave = tf.cast(pre_emph(wave, preemph), tf.float32)
        noisy = tf.cast(pre_emph(noisy, preemph), tf.float32)
        ref =  tf.cast(pre_emph(ref, preemph), tf.float32)

    return wave, noisy, ref
