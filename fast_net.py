#!/usr/bin/env python

import os
from os.path import exists, join, split

import tensorflow as tf
import numpy as np

from PIL import Image

import pdb


CITYSCAPE_PALLETE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


def train(sess, params, data):
    # unpack params
    seg_type = params['seg_type']
    seg_dim = params['seg_dim']
    out_type = params['out_type']
    out_dim = params['out_dim']

    # build graph
    x = tf.placeholder(seg_type, [None, seg_dim])

    hid = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=hid, units=out_dim, activation=None)

    y_ = tf.placeholder(out_type, [None, out_dim])

    # define op
    loss = tf.reduce_mean(tf.squared_difference(y_, y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # extract data
    data_x = data['x']
    data_y = data['y']

    data_size = len(data_x)
    batch_size = 10

    tf.global_variables_initializer().run()
    for i in range(int(data_size / batch_size)):
        lo =     i * batch_size
        hi = (i+1) * batch_size
        loss_val, train_err = sess.run([loss, train_step],
            feed_dict={
                x:  data_x[lo : hi],
                y_: data_y[lo : hi]
            }
        )

    return (x, y)

def run(sess, ops, data):

    x, y = ops
    y_val = sess.run(y, feed_dict={
        x: data['x']
    })
    y_val = y_val.astype(int)

    return y_val

def get_lists(datadir, phase):

    input_path = join(datadir, phase + '_inputs.txt')
    label_path = join(datadir, phase + '_labels.txt')

    input_list = [line.strip() for line in open(input_path, 'r')]
    label_list = [line.strip() for line in open(label_path, 'r')] if exists(label_path) else []

    return (input_list, label_list)

def load_files(filenames, datadir):
    """
    Read a batch of files into a (B x C x H x W) tensor.
    """
    out = np.empty([])
    for ind in range(len(filenames)):
        fn = os.path.join(datadir, filenames[ind])
        arr = np.asarray(Image.open(fn))
        arr = np.expand_dims(arr, axis=0)
        if ind == 0:
            out = arr
        else:
            out = np.append(out, arr, axis=0)
    return out

def save_files(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind])
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_files_color(predictions, filenames, output_dir, palettes):
    """
    Saves a given (B x C x H x W) tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def main():

    input_files, label_files = get_lists('data-2', 'train')
    test_input_files, _ = get_lists('data-2', 'test')

    inputs = load_files(input_files, 'data-2')
    labels = load_files(label_files, 'data-2')
    test_inputs = load_files(test_input_files, 'data-2')

    train_data = {'x': inputs.reshape((inputs.shape[0], -1)),
                  'y': labels.reshape((labels.shape[0], -1))}
    test_data  = {'x': test_inputs.reshape((test_inputs.shape[0], -1))}

    params = {'seg_type': tf.float32,
              'seg_dim' : 1 * 1024 * 2048,
              'out_type': tf.float32,
              'out_dim' : 1 * 1024 * 2048}

    with tf.Session() as sess:
        ops = train(sess=sess, params=params, data=train_data)
        pred = run(sess=sess, ops=ops, data=test_data)
        pred = pred.reshape((-1, inputs.shape[1], inputs.shape[2]))
        save_files(pred, test_input_files, 'pred-2')
        save_files_color(pred, test_input_files, 'pred-2_color', CITYSCAPE_PALLETE)

if __name__ == '__main__':
    main()

