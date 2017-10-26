""" scene evaluation
    original code: tensorflow cifar-10 tutorial.

    moving average, use std 10crop eval, softmax-linear, mean.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import json
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint
import numpy as np

import scene_input
import scene
import scene_resnet as resnet
# import scene_vgg as vgg
import utility_tensorflow as utils
import tqdm

root = "/scratch/lyc/ai_challenger_scene/"

data_dir = os.path.join(root, "data")
eval_dir = os.path.join(root, "tmp/eval")
checkpoint_dir = os.path.join(root, "tmp/checkpoint")
model_dir = os.path.join(root, "model")
print(checkpoint_dir)
print(data_dir)

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
            logits = scene.inference(images, keep_prob)

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


def get_accuracy_10crop_resnet(sess, top_1_op, top_3_op, data_init_op, is_training):
    """ get top1 and top3 accuracy for validation set. """
    # Initialize the correct dataset
    sess.run(data_init_op)
    num_correct, num_correct_top3 = 0, 0
    for i in tqdm.tqdm(range(num_examples_per_epoch_for_val)):
        try:
            correct_pred, correct_pred_top3 = sess.run(
                [top_1_op, top_3_op], {is_training: False})
            num_correct += correct_pred.sum()
            num_correct_top3 += correct_pred_top3.sum()
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_examples_per_epoch_for_val
    acc_top3 = float(num_correct_top3) / num_examples_per_epoch_for_val
    return acc, acc_top3


def evaluate_10crop():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        keep_prob = tf.placeholder(tf.float32)

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            logits = scene.inference(images_10crop_batched, keep_prob)
            prob_10crop = tf.reduce_mean(logits, axis=0)

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


def evaluate_10crop_resnet():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            logits = scene.inference_resnet(images_10crop_batched, is_training)
            prob_10crop = tf.reduce_mean(logits, axis=0)

        # Calculate predictions.
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_10crop = tf.reshape(prob_10crop, [-1, scene_input.num_classes])
        label = tf.reshape(label, [-1])
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        top_1_op = tf.nn.in_top_k(prob_10crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_10crop, label, 3)

        # Restore not trainable variables
        # list_all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # list_all_trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print("len(list_all_variables): {}".format(len(list_all_variables)))
        # for v in list_all_variables:
        #     print(v)
        # print("len(list_all_trainable_variables): {}".format(len(list_all_trainable_variables)))
        # for v in list_all_trainable_variables:
        #     print(v)
        # variables_to_restore = {}
        # for v in list_all_variables:
        #     if v not in list_all_trainable_variables:
        #         variables_to_restore[v.name] = v

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        print("len(variables_to_restore): {}".format(len(variables_to_restore)))
        # for v in variables_to_restore.keys():
        #     print("{} --> {}".format(v, variables_to_restore[v]))
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("variables in checkpoint file {}".format(ckpt.model_checkpoint_path))
                utils.print_tensors_in_checkpoint_file(
                    file_name=ckpt.model_checkpoint_path, tensor_name=None, all_tensors=True)
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                raise ValueError("Cannot find checkpoint data in {}".format(checkpoint_dir))

            # accuracy on validation set.
            acc_top1, acc_top3 = get_accuracy_10crop_resnet(sess, top_1_op, top_3_op,
                                                            val_data_init_op, is_training)
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))


def main(argv=None):  # pylint: disable=unused-argument
    # evaluate()
    evaluate_10crop_resnet()


if __name__ == '__main__':
    tf.app.run()
