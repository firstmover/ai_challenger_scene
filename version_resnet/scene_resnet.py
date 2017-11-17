""" resnet """

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import datetime

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import skimage.io
import skimage.transform
import numpy as np
import time

from config import Config
from synset import *
import utility_tensorflow as utils

import scene_config as c

data_dir = c.data_dir
model_dir = c.model_dir

BN_DECAY = c.BN_DECAY
BN_EPSILON = c.BN_EPSILON
CONV_WEIGHT_DECAY = c.CONV_WEIGHT_DECAY
CONV_WEIGHT_STDDEV = c.CONV_WEIGHT_STDDEV
FC_WEIGHT_DECAY = c.FC_WEIGHT_DECAY
FC_WEIGHT_STDDEV = c.FC_WEIGHT_STDDEV

input_size = c.input_size

activation = c.activation

RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
RESNET_FC_VARIABLES = 'resnet_fc_variables'
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838]
IMAGENET_MEAN_RGB = [123.151630838, 115.902882574, 103.062623801]


def inference(x, is_training,
              num_classes=1000,
              num_blocks=None,  
              use_bias=False,  # defaults to using batch norm
              bottleneck=True,
              freeze_bn=False):

    # assume x is rgb, not scaled, mean subtracted
    x = x * 1.0

    # Convert RGB to BGR
    # already subtracted mean value in image preproccess
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    x = tf.concat(axis=3, values=[blue, green, red])
    assert x.get_shape().as_list()[1:] == [224, 224, 3]

    # defaults to 50-layer network
    if num_blocks is None:
        num_blocks = [3, 4, 6, 3]

    c = Config()
    c['bottleneck'] = bottleneck
    c['freeze_bn'] = freeze_bn
    # print("inferece bottleneck: {}".format(c['bottleneck']))
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('scale4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('scale5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)

    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return x


def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    # print("block bottleneck: {}".format(c['bottleneck']))
    if c['bottleneck']:
        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)

    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    if c['freeze_bn']:
        beta = _get_variable(
            'beta', params_shape, 
            initializer=tf.zeros_initializer,
            trainable=False)
        gamma = _get_variable(
            'gamma',  params_shape, 
            initializer=tf.ones_initializer,
            trainable=False)

        moving_mean = _get_variable(
            'moving_mean', params_shape, 
            initializer=tf.zeros_initializer, 
            trainable=False)
        moving_variance = _get_variable(
            'moving_variance', params_shape, 
            initializer=tf.ones_initializer, 
            trainable=False)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, tf.no_op())
        mean, variance = moving_mean, moving_variance
    else:
        beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
        gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

        moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
        moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(c['is_training'], 
            lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    # x.set_shape(inputs.get_shape()) ??

    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    tf.add_to_collection(name=RESNET_FC_VARIABLES, value=weights)
    tf.add_to_collection(name=RESNET_FC_VARIABLES, value=biases)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    """A little wrapper around tf.get_variable to do weight decay and add to"""
    """resnet collection"""
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def get_variables_except_fc():
    all_variables = tf.get_collection(RESNET_VARIABLES)
    fc_variables = tf.get_collection(RESNET_FC_VARIABLES)
    list_v = [x for x in all_variables if x not in fc_variables]
    print("len(all_variables): {}".format(len(all_variables)))
    print("len(fc_variables): {}".format(len(fc_variables)))
    print("len(list_v): {}".format(len(list_v)))
    return list_v


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, size=224):
    # img here is [0, 256]
    img = skimage.io.imread(path)
    # print("row image matrix:")
    # print(img)
    # print("max: {} min: {}".format(np.max(img), np.min(img)))
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    # resized image is [0, 1]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    # print("processed image:")
    # print(resized_img)
    # print("max: {} min: {}".format(np.max(resized_img), np.min(resized_img)))
    return resized_img


# returns the top1 string
def print_prob(prob):
    #print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: {}".format(top1))
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("Top5: {}".format(top5))
    return top1


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def main():
    """ load pretained model, ImageNet. 
        Get exact the same answer as forward.py.
        Remember check the preprocessing.
    """

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        rgb_scaled = tf.placeholder(tf.float32, [None, 224, 224, 3], "image")

        # preprocess. ImageNet: bgr, not scaled, subtract mean
        rgb = rgb_scaled * 256.0
        means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 1, 3])
        rgb = rgb - means
        with tf.name_scope('inference'):
            logits = inference(
                rgb, 
                num_classes=1000, 
                is_training=is_training, 
                bottleneck=True, 
                num_blocks=None)

        init_var_op = tf.global_variables_initializer()

        list_resnet_variables = tf.get_collection(RESNET_VARIABLES)
        print("type(list_resnet_variables): {}".format(type(list_resnet_variables)))
        print("resnet variables number: {}".format(len(list_resnet_variables)))
        # for v in list_resnet_variables:
        #     print(v)
        saver = tf.train.Saver(list_resnet_variables)

        with tf.Session() as sess:
            sess.run(init_var_op)

            pretrained_model_path = os.path.join(model_dir, checkpoint_fn(layer))
            if os.path.exists(pretrained_model_path):
                # Restores from checkpoint
                saver.restore(sess, pretrained_model_path)
                print("variables in checkpoint file {}".format(pretrained_model_path))
                utils.print_tensors_in_checkpoint_file(
                    file_name=pretrained_model_path, tensor_name=None, all_tensors=True)
            else:
                raise ValueError(
                    "Cannot find pretrained model: {}".format(pretrained_model_path))

            img = load_image(os.path.join(data_dir, "cat.jpg"))
            print("img: ")
            print(img)
            batch_images = img.reshape((-1, 224, 224, 3))

            l, _rgb = sess.run([logits, rgb], feed_dict={rgb_scaled:batch_images, is_training:False})
            print("rgb image matrix: ")
            print(_rgb)
            print_prob(l[0])


if __name__ == "__main__":
    main()