""" scene test
    moving average, use std 10crop eval, softmax-linear, mean.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import scene_input
import scene
import scene_resnet as resnet
# import scene_vgg as vgg
import utility_tensorflow as utils
import tqdm

import scene_config as c

data_dir = c.data_dir
tmp_dir = c.tmp_dir
checkpoint_dir = c.checkpoint_dir
model_dir = c.model_dir
test_image_inference_file = c.test_image_inference_file
num_examples_per_epoch_for_test_1 = c.num_examples_per_epoch_for_test_1


def test_10crop():
    test_filenames = scene_input.list_images_test()

    with tf.Graph().as_default() as g:
        is_training = tf.placeholder(tf.bool, [], "is_training")

        images_10crop_batched, filename, test_data_init_op \
            = scene_input.get_dataset_10crop_test(test_filenames)

        with tf.name_scope('inference'):
            logits = scene.inference_resnet(images_10crop_batched, is_training, layer=101)
            prob_10crop = tf.reduce_mean(logits, axis=0)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            scene.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver_with_moving_average = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            restore_epoch = "13"
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
            sess.run(test_data_init_op)
            print("writing test image inference to {}".format(test_image_inference_file))
            with open(test_image_inference_file, "w") as f:
                for i in tqdm.tqdm(range(num_examples_per_epoch_for_test_1)):
                    try:
                        name, _prob_10crop = sess.run([filename, prob_10crop], {is_training: False})
                        sorted_index = np.argsort(_prob_10crop)
                        top3_label = sorted_index[-3:]
                        name = os.path.basename(name)
                        inference_info = "{} {} {} {}\n".format(
                            name, top3_label[2], top3_label[1], top3_label[0])
                        f.write(inference_info)
                    except tf.errors.OutOfRangeError:
                        break
            

def main(argv=None):  # pylint: disable=unused-argument
    test_10crop()


if __name__ == '__main__':
    tf.app.run()
