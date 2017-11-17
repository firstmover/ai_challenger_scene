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

import utility_tensorflow as utils
import tqdm

import scene_config as c
import scene
import scene_input

data_dir = c.data_dir
tmp_dir = c.tmp_dir
checkpoint_dir = c.checkpoint_dir
model_dir = c.model_dir
wrong_image_info_file = c.wrong_image_info_file

num_examples_per_epoch_for_train = c.num_examples_per_epoch_for_train
num_examples_per_epoch_for_val = c.num_examples_per_epoch_for_val

batch_size = c.batch_size


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


def evaluate_10crop_resnet():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        images_10crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_10crop_eval(val_filenames, val_labels)

        with tf.name_scope('inference'):
            logits = scene.inference_resnet(images_10crop_batched, is_training, layer=c.resnet_layer)
            prob_10crop = tf.reduce_mean(logits, axis=0)

        # Calculate predictions.
        print("prob_10crop.shape: {}".format(prob_10crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_10crop = tf.reshape(prob_10crop, [-1, c.num_classes])
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
        # print("len(variables_to_restore): {}".format(len(variables_to_restore)))
        # for v in variables_to_restore.keys():
        #     print("{} --> {}".format(v, variables_to_restore[v]))
        saver_with_moving_average = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            list_acc_top1_global, list_acc_top3_global = [], []
            list_acc_top1_moving_average, list_acc_top3_moving_average = [], [] 
            list_enumerate_epoch = [10, 11, 12, 13, 14]
            for n_epoch in list_enumerate_epoch:
                restore_epoch = "{}".format(n_epoch)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                restore_checkpoint_path = os.path.join(checkpoint_dir, "scene-" + restore_epoch)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    restore_checkpoint_path = ckpt.model_checkpoint_path
                elif not tf.train.checkpoint_exists(restore_checkpoint_path):
                    raise ValueError("Cannot find checkpoint file: {} or in dir: {}".format(
                        restore_checkpoint_path, checkpoint_dir))
                    
                # print("variables in checkpoint file {}".format(restore_checkpoint_path))
                # utils.print_tensors_in_checkpoint_file(
                #     file_name=restore_checkpoint_path, tensor_name=None, all_tensors=True)            
                print("restore checkpoint from file: {}".format(restore_checkpoint_path))

                # print("restore global variables.")
                # saver.restore(sess, restore_checkpoint_path)
                # accuracy on validation set.
                # acc_top1, acc_top3 = get_accuracy_10crop_resnet(sess, top_1_op, top_3_op,
                #                                                 val_data_init_op, is_training)
                # print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))
                # list_acc_top1_global.append(acc_top1)
                # list_acc_top3_global.append(acc_top3)

                print("restore global variables with moving average.")
                saver_with_moving_average.restore(sess, restore_checkpoint_path)
                # accuracy on validation set.
                acc_top1, acc_top3 = get_accuracy_10crop_resnet(sess, top_1_op, top_3_op,
                                                                val_data_init_op, is_training)
                print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))
                list_acc_top1_moving_average.append(acc_top1)
                list_acc_top3_moving_average.append(acc_top3)
            # for i in range(30):
            #     print("{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(
            #         list_acc_top1_global[i], list_acc_top3_global[i], 
            #         list_acc_top1_moving_average[i], list_acc_top3_moving_average[i]))


def evaluate_90crop_resnet():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        images_90crop_batched, label, val_data_init_op \
            = scene_input.get_dataset_90crop_eval(val_filenames, val_labels)
        print("images_90crop_batched.shape: {}".format(images_90crop_batched.shape))

        with tf.name_scope('inference'):
            logits = scene.inference_resnet(images_90crop_batched, is_training, layer=c.resnet_layer)
            prob_90crop = tf.reduce_mean(logits, axis=0)

        # Calculate predictions.
        print("prob_90crop.shape: {}".format(prob_90crop.shape))
        print("label.shape: {}".format(label.shape))
        prob_90crop = tf.reshape(prob_90crop, [-1, c.num_classes])
        label = tf.reshape(label, [-1])
        print("prob_90crop.shape: {}".format(prob_90crop.shape))
        print("label.shape: {}".format(label.shape))
        top_1_op = tf.nn.in_top_k(prob_90crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_90crop, label, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver_with_moving_average = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            list_acc_top1_moving_average, list_acc_top3_moving_average = [], [] 
            list_enumerate_epoch = range(10, 20, 1)
            for n_epoch in list_enumerate_epoch:
                restore_epoch = "{}".format(n_epoch)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                restore_checkpoint_path = os.path.join(checkpoint_dir, "scene-" + restore_epoch)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    restore_checkpoint_path = ckpt.model_checkpoint_path
                elif not tf.train.checkpoint_exists(restore_checkpoint_path):
                    raise ValueError("Cannot find checkpoint file: {} or in dir: {}".format(
                        restore_checkpoint_path, checkpoint_dir))
                            
                print("restore checkpoint from file: {}".format(restore_checkpoint_path))
                print("restore global variables with moving average.")
                saver_with_moving_average.restore(sess, restore_checkpoint_path)
                # accuracy on validation set.
                acc_top1, acc_top3 = get_accuracy_10crop_resnet(sess, top_1_op, top_3_op,
                                                                val_data_init_op, is_training)
                print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc_top1, acc_top3))
                list_acc_top1_moving_average.append(acc_top1)
                list_acc_top3_moving_average.append(acc_top3)
            for i in range(len(list_enumerate_epoch)):
                print("{0:} {1:.4f} {2:.4f}".format(list_enumerate_epoch[i],
                    list_acc_top1_moving_average[i], list_acc_top3_moving_average[i]))


def evaluate_10crop_resnet_with_check():
    val_filenames, val_labels = scene_input.list_images('validation')

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        images_10crop_batched, label, filename, val_data_init_op \
            = scene_input.get_dataset_10crop_eval_with_filename(val_filenames, val_labels)

        with tf.name_scope('inference'):
            logits = scene.inference_resnet(images_10crop_batched, is_training)
            prob_10crop = tf.reduce_mean(logits, axis=0)

        # Calculate predictions.
        prob_10crop = tf.reshape(prob_10crop, [-1, scene_input.num_classes])
        label = tf.reshape(label, [-1])
        top_1_op = tf.nn.in_top_k(prob_10crop, label, 1)
        top_3_op = tf.nn.in_top_k(prob_10crop, label, 3)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver_with_moving_average = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            restore_epoch = "23"
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            restore_checkpoint_path = os.path.join(checkpoint_dir, "scene-" + restore_epoch)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                restore_checkpoint_path = ckpt.model_checkpoint_path
            elif not tf.train.checkpoint_exists(restore_checkpoint_path):
                raise ValueError("Cannot find checkpoint file: {} or in dir: {}".format(
                    restore_checkpoint_path, checkpoint_dir))
            print("restore checkpoint from file: {}".format(restore_checkpoint_path))
            print("restore global variables with moving average.")
            saver_with_moving_average.restore(sess, restore_checkpoint_path)

            # accuracy on validation set.
            sess.run(val_data_init_op)
            num_correct, num_correct_top3 = 0, 0
            print("writing wrong image info to {}".format(wrong_image_info_file))
            with open(wrong_image_info_file, "w") as f:
                for i in tqdm.tqdm(range(num_examples_per_epoch_for_val)):
                    try:
                        correct_pred, correct_pred_top3, name, _prob_10crop, _label = sess.run(
                            [top_1_op, top_3_op, filename, prob_10crop, label], {is_training: False})
                        num_correct += correct_pred.sum()
                        num_correct_top3 += correct_pred_top3.sum()
                        if correct_pred_top3.sum() == 0:
                            # filename, top3 inference, label
                            _prob_10crop = _prob_10crop[0]
                            _label = _label[0]
                            sorted_index = np.argsort(_prob_10crop)
                            top3_label = sorted_index[-3:]
                            wrong_image_info = "{} {} {} {} {}\n".format(
                                name, top3_label[2], top3_label[1], top3_label[0], _label)
                            f.write(wrong_image_info)
                    except tf.errors.OutOfRangeError:
                        break
            # Return the fraction of datapoints that were correctly classified
            acc = float(num_correct) / num_examples_per_epoch_for_val
            acc_top3 = float(num_correct_top3) / num_examples_per_epoch_for_val
            print('Val: accuracy (top1){0:.4f} (top3){1:.4f}'.format(acc, acc_top3))


def check_wrong_inferenced_images():
    with open(wrong_image_info_file) as f:
        wrong_info = f.read().splitlines()
        print("len(wrong_info): {}".format(len(wrong_info)))
        print(wrong_info[0])
            

def main(argv=None):  # pylint: disable=unused-argument
    # evaluate()
    evaluate_90crop_resnet()


if __name__ == '__main__':
    tf.app.run()
