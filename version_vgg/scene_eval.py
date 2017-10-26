""" scene evaluation
    original code: tensorflow cifar-10 tutorial.

    experiment results:
    1.  random scale moving average, val acc 0.9181.
        random scale moving average, standard 10 crop, val acc
        0.8944: (softmax resize-224 mean)
        0.8962: (softmax resize-256 mean)
        0.9044: (softmax-linear resize-256 mean)
        0.8942: (softmax-linear resize-256 max)
        0.8831: (softmax-linear resize-256 mean crop-center-image
                => means your code is wrong.)
        corrected val acc:
        0.9392: (softmax-linear resize-256 mean)
        0.9308: (softmax-linear resize-256 max)
        0.9350: (softmax resize-256 mean)
        0.9309: (softmax resize-256 max)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import json
import tensorflow as tf
import numpy as np

import scene_input
import scene
import scene_vgg as vgg
import utility_tensorflow as utils
import tqdm

data_dir = "/scratch/lyc/ai_challenger_scene/data"
eval_dir = '/scratch/lyc/ai_challenger_scene/tmp/eval'
checkpoint_dir = "/scratch/lyc/ai_challenger_scene/tmp/checkpoint"
model_dir = "/scratch/lyc/ai_challenger_scene/model"

num_examples_per_epoch_for_train = scene_input.num_examples_per_epoch_for_train
num_examples_per_epoch_for_val = scene_input.num_examples_per_epoch_for_val

batch_size = 64


def get_accuracy(sess, top_1_op, top_3_op, data_init_op, keep_prob):
    """ get top1 and top3 accuracy for validation set. """
    # Initialize the correct dataset
    sess.run(data_init_op)
    num_correct, num_correct_top3, num_samples = 0, 0, 0
    num_steps_per_epoch = int(num_examples_per_epoch_for_val / batch_size) + 1
    for i in tqdm.tqdm(range(num_steps_per_epoch)):
        try:
            correct_pred, correct_pred_top3 = sess.run(
                [top_1_op, top_3_op], {keep_prob: 1.0})
            num_correct += correct_pred.sum()
            num_correct_top3 += correct_pred_top3.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    acc_top3 = float(num_correct_top3) / num_samples
    return acc, acc_top3


def get_accuracy_10crop(sess, top_1_op, top_3_op, data_init_op, keep_prob):
    """ get top1 and top3 accuracy for validation set. """
    # Initialize the correct dataset
    sess.run(data_init_op)
    num_correct, num_correct_top3 = 0, 0
    for i in tqdm.tqdm(range(num_examples_per_epoch_for_val)):
        try:
            correct_pred, correct_pred_top3 = sess.run(
                [top_1_op, top_3_op], {keep_prob: 1.0})
            num_correct += correct_pred.sum()
            num_correct_top3 += correct_pred_top3.sum()
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_examples_per_epoch_for_val
    acc_top3 = float(num_correct_top3) / num_examples_per_epoch_for_val
    return acc, acc_top3


def evaluate():
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = scene_input.list_images('train')
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        keep_prob = tf.placeholder(tf.float32)

        images, labels, train_data_init_op, val_data_init_op \
            = scene_input.get_dataset(train_filenames, train_labels,
                                      val_filenames, val_labels, batch_size)

        with tf.name_scope('inference'):
            conv_net = vgg.Vgg16()
            conv_net.build(images, keep_prob, scene_input.num_classes)
            logits = conv_net.get_softmax_linear()

        # Calculate predictions.
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_3_op = tf.nn.in_top_k(logits, labels, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                raise ValueError("Cannot find checkpoint data in {}".format(checkpoint_dir))

            # accuracy on validation set.
            acc_top1, acc_top3 = get_accuracy(sess, top_1_op, top_3_op,
                                              val_data_init_op, keep_prob)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))


def evaluate_10crop_1():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        keep_prob = tf.placeholder(tf.float32)

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            conv_net = vgg.Vgg16()
            conv_net.build(images_10crop_batched, keep_prob, scene_input.num_classes)
            prob_10crop = tf.reduce_max(conv_net.get_softmax_linear(), axis=0)

        # Calculate predictions.
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_10crop = tf.reshape(prob_10crop, [-1, scene_input.num_classes])
        label = tf.reshape(label, [-1])
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        top_1_op = tf.nn.in_top_k(prob_10crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_10crop, label, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                raise ValueError("Cannot find checkpoint data in {}".format(checkpoint_dir))

            # accuracy on validation set.
            acc_top1, acc_top3 = get_accuracy_10crop(sess, top_1_op, top_3_op,
                                                     val_data_init_op, keep_prob)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))


def evaluate_10crop_2():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        keep_prob = tf.placeholder(tf.float32)

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            conv_net = vgg.Vgg16()
            conv_net.build(images_10crop_batched, keep_prob, scene_input.num_classes)
            prob_10crop = tf.reduce_mean(conv_net.get_softmax(), axis=0)

        # Calculate predictions.
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_10crop = tf.reshape(prob_10crop, [-1, scene_input.num_classes])
        label = tf.reshape(label, [-1])
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        top_1_op = tf.nn.in_top_k(prob_10crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_10crop, label, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                raise ValueError("Cannot find checkpoint data in {}".format(checkpoint_dir))

            # accuracy on validation set.
            acc_top1, acc_top3 = get_accuracy_10crop(sess, top_1_op, top_3_op,
                                                     val_data_init_op, keep_prob)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))


def evaluate_10crop_3():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        keep_prob = tf.placeholder(tf.float32)

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            conv_net = vgg.Vgg16()
            conv_net.build(images_10crop_batched, keep_prob, scene_input.num_classes)
            prob_10crop = tf.reduce_max(conv_net.get_softmax(), axis=0)

        # Calculate predictions.
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_10crop = tf.reshape(prob_10crop, [-1, scene_input.num_classes])
        label = tf.reshape(label, [-1])
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        top_1_op = tf.nn.in_top_k(prob_10crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_10crop, label, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                raise ValueError("Cannot find checkpoint data in {}".format(checkpoint_dir))

            # accuracy on validation set.
            acc_top1, acc_top3 = get_accuracy_10crop(sess, top_1_op, top_3_op,
                                                     val_data_init_op, keep_prob)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))


def main(argv=None):  # pylint: disable=unused-argument
    # evaluate()
    # evaluate_10crop_1()
    # evaluate_10crop_2()
    evaluate_10crop_3()


if __name__ == '__main__':
    tf.app.run()
