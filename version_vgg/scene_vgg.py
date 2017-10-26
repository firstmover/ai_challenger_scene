""" vgg16 / 19 model """

import os

import tensorflow as tf
import numpy as np

# VGG_MEAN = [103.939, 116.779, 123.68]

model_dir = '/scratch/lyc/ai_challenger_scene/model'


class Vgg16:
    """
    A trainable version VGG16.
    """

    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            vgg16_npy_path = os.path.join(model_dir, 'vgg16.npy')
        self._data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        self._var_dict = {}

    def build(self, rgb, keep_prob, num_classes):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255.0]
        :param keep_prob: drop out possibility, placeholder
        :param num_classes:
        """

        rgb_scaled = rgb * 1.0

        # Convert RGB to BGR
        # already subtracted mean value in image preproccess
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[blue, green, red])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self._conv_layer(bgr, 3, 64, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')

        self.fc6 = self._fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, keep_prob)

        self.fc7 = self._fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        self.relu7 = tf.nn.dropout(self.relu7, keep_prob)

        self.fc8 = self._fc_layer(self.relu7, 4096, num_classes, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        if self._data_dict is not None:
            del self._data_dict

    def get_softmax_linear(self):
        return self.fc8

    def get_softmax(self):
        return self.prob

    def get_DNN_variables(self):
        v_names = [("fc6", 0), ("fc6", 1), ("fc7", 0), ("fc7", 1), ("fc8", 0), ("fc8", 1)]
        return [self._var_dict[name] for name in v_names]

    def _avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self._get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def _fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self._get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self._get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self._get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def _get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self._get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self._get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def _get_var(self, initial_value, name, idx, var_name):
        if self._data_dict is not None and name in self._data_dict:
            # initiate fc layer variables with truncated normal
            if name in ["fc6", "fc7", "fc8"]:
                value = initial_value
            else:
                value = self._data_dict[name][idx]
        else:
            value = initial_value

        var = tf.get_variable(initializer=value, name=var_name)

        self._var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var
