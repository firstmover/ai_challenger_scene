""" scene classification configuration. """
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

import numpy as np 
import tensorflow as tf

""" path """

root = "/home/lyc/ai_challenger_scene/"

json_train = "ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json"
image_file_train = "ai_challenger_scene_train_20170904/scene_train_images_20170904"
json_validation = "ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json"
image_file_validation = "ai_challenger_scene_validation_20170908/scene_validation_images_20170908"
test_filename = "ai_challenger_scene_test_1"

data_dir = os.path.join(root, "data")
tmp_dir = os.path.join(root, "tmp_101")
log_dir = os.path.join(tmp_dir, "log")
checkpoint_dir = os.path.join(tmp_dir, "checkpoint")
model_dir = os.path.join(root, "model")

acc_loss_path = os.path.join(tmp_dir, "acc_loss.txt")
wrong_image_info_file = os.path.join(tmp_dir, "wrong_image_info.txt")
test_image_inference_file = os.path.join(tmp_dir, "test_image_inference.txt")



""" train routine """

# dropout_keep_prob = 0.5
batch_size = 128

num_epochs = 30 # 1000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.

# learning rate decay constant
NUM_EPOCHS_PER_DECAY = 50 # 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# momentum SGD optimizer
initial_learning_rate_mSGD = 0.01
momentum = 0.9

# write tensorboard info every freq steps
tensorboard_write_frq = 200

# checkpoint every freq epoch
checkpoint_freq = 1

use_minimal_summary = True



""" model architecture """

resnet_layer = 50

num_workers = 16

BN_DECAY = 0.9997
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01

input_size = 224

activation = tf.nn.relu



""" input and augmentation """

rotate_angle = 45

num_classes = 80
num_examples_per_epoch_for_train = 53879
num_examples_per_epoch_for_val = 7120
num_examples_per_epoch_for_test_1 = 7040