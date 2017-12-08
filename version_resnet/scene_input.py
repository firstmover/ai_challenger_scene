""" scene data reader version 2 """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import json
import tensorflow as tf
import numpy as np
import scipy.misc
import math
import tqdm

from tensorflow.contrib.image.python.ops import image_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import control_flow_ops

import scene_config as c

data_dir = c.data_dir

json_train = c.json_train
image_file_train = c.image_file_train
json_validation = c.json_validation
image_file_validation = c.image_file_validation
test_filename = c.test_filename

num_workers = c.num_workers

rotate_angle = c.random_rotate_range

num_classes = c.num_classes
num_examples_per_epoch_for_train = c.num_examples_per_epoch_for_train
num_examples_per_epoch_for_val = c.num_examples_per_epoch_for_val
num_examples_per_epoch_for_test_1 = c.num_examples_per_epoch_for_test_1

IMAGENET_MEAN_RGB = [123.151630838, 115.902882574, 103.062623801]
# vgg_mean = [123.68, 116.78, 103.94]

input_size = c.input_size


def list_images(data_set_type):
    """ Get all the images and labels. """
    if data_set_type == "train":
        json_file_path = os.path.join(data_dir, json_train)
        image_dir = os.path.join(data_dir, image_file_train)
        num = num_examples_per_epoch_for_train
    elif data_set_type == "validation":
        json_file_path = os.path.join(data_dir, json_validation)
        image_dir = os.path.join(data_dir, image_file_validation)
        num = num_examples_per_epoch_for_val
    else:
        raise ValueError("Wrong data set type. Use train or validation.")

    print("reading json file from: {}".format(json_file_path))
    image_label = None
    with open(json_file_path, "r") as f:
        for line in f:
            image_label = json.loads(line)
    image_number = len(image_label)
    assert image_number == num
    print("image number: {}".format(image_number))

    files_labels = []
    for i in range(image_number):
        image_name, label = image_label[i]["image_id"], image_label[i]["label_id"]
        files_labels.append((os.path.join(image_dir, image_name), int(label)))

    filenames, labels = zip(*files_labels)
    filenames = list(filenames)
    labels = list(labels)

    return filenames, labels


def list_images_test():
    test_file_path = os.path.join(data_dir, test_filename)
    print("Get image names in dir: {}".format(test_file_path))
    image_names = os.listdir(test_file_path)
    image_paths = []
    for name in image_names:
        image_paths.append(os.path.join(test_file_path, name))
    return image_paths


def list_images_testb():
    test_file_path = os.path.join(data_dir, c.testb_filename)
    print("Get image names in dir: {}".format(test_file_path))
    image_names = os.listdir(test_file_path)
    image_paths = []
    for name in image_names:
        image_paths.append(os.path.join(test_file_path, name))
    return image_paths


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = input_size * 256.0 / 224
    height, width = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height, new_width = tf.to_int32(height * scale), tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


def _parse_function_with_random_scale(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    # get random side from [256, 480].
    # if input_size > 256, set lower bound to input_size
    # use [256, 320], performance decrease, setting see result.txt
    # resize_range = [256, 480]
    resize_range = [c.random_scale_lower, c.random_scale_upper]
    if resize_range[0] < input_size:
        resize_range[0] = input_size
    if resize_range[1] < resize_range[0]:
        raise ValueError("Wrong random scale range.")
    smallest_side = resize_range[0] + \
        (resize_range[1] - resize_range[0]) * random_ops.random_uniform([1], 0, 1)[0]
    height, width = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height, new_width = tf.to_int32(height * scale), tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


def get_dataset_with_random_scale(
    train_filenames, train_labels, val_filenames, val_labels, batch_size):
    def training_preprocess(image, label):
        crop_image = tf.random_crop(image, [input_size, input_size, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)

        means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
        centered_image = flip_image - means

        return centered_image, label

    def val_preprocess(image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, input_size, input_size)

        means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
        centered_image = crop_image - means

        return centered_image, label

    # Training dataset
    train_filenames, train_labels = tf.constant(train_filenames), tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function_with_random_scale,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.map(training_preprocess,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(batch_size)

    # Validation dataset
    val_filenames, val_labels = tf.constant(val_filenames), tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(_parse_function,
                                  num_threads=num_workers, output_buffer_size=batch_size)
    val_dataset = val_dataset.map(val_preprocess,
                                  num_threads=num_workers, output_buffer_size=batch_size)

    batched_val_dataset = val_dataset.batch(batch_size)

    iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                       batched_train_dataset.output_shapes)
    images, labels = iterator.get_next()

    train_data_init_op = iterator.make_initializer(batched_train_dataset)
    val_data_init_op = iterator.make_initializer(batched_val_dataset)

    return images, labels, train_data_init_op, val_data_init_op


def image_crop(batched_image, offset, side):
    # crop l * l, 3 from batch_image, use offset(center point)
    patch = tf.image.extract_glimpse(batched_image, [side, side],
        offsets=tf.to_float([offset]), centered=False, normalized=False)
    return tf.reshape(patch, [side, side, 3])


def crop5(image, height, width, side):
    batched_image = tf.reshape(image, [-1, height, width, 3])

    half_height, half_width = tf.to_int32(height / 2), tf.to_int32(width / 2)
    half_side = int(side / 2)
    mid = image_crop(batched_image, [half_height, half_width], side)
    l_u = image_crop(batched_image, [half_side, half_side], side)
    l_d = image_crop(batched_image, [height - half_side, half_side], side)
    r_u = image_crop(batched_image, [half_side, width - half_side], side)
    r_d = image_crop(batched_image, [height - half_side, width - half_side], side)

    return tf.stack([mid, l_u, l_d, r_u, r_d])

def _crop_along_h(image, height, width, side):
	# crop 3 * 5, side * side image along height
    batched_image = tf.reshape(image, [-1, height, width, 3])
    half_height, half_width = tf.to_int32(height / 2), tf.to_int32(width / 2)
    u = image_crop(batched_image, [half_width, half_width], width)
    b_u = crop5(u, width, width, side)
    m = image_crop(batched_image, [half_height, half_width], width)
    b_m = crop5(m, width, width, side)
    d = image_crop(batched_image, [height - half_width, half_width], width)
    b_d = crop5(d, width, width, side)
    return tf.concat([b_u, b_m, b_d], 0)

def _crop_along_w(image, height, width, side):
	# crop 3 * 5, side * side image along width
    batched_image = tf.reshape(image, [-1, height, width, 3])
    half_height, half_width = tf.to_int32(height / 2), tf.to_int32(width / 2)
    l = image_crop(batched_image, [half_height, half_height], height)
    b_l = crop5(l, height, height, side)
    m = image_crop(batched_image, [half_height, half_width], height)
    b_m = crop5(m, height, height, side)
    r = image_crop(batched_image, [half_height, width - half_height], height)
    b_r = crop5(r, height, height, side)
    return tf.concat([b_l, b_m, b_r], 0)


def val_preprocess_standard_10crop_eval(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = input_size * 256 / 224.0 # keep scale constant
    height, width = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height, new_width = tf.to_int32(height * scale), tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
    centered_image = resized_image - means
    flipped_image = tf.image.flip_left_right(centered_image)

    return tf.concat([crop5(centered_image, new_height, new_width, input_size),
                      crop5(flipped_image, new_height, new_width, input_size)], 0), label


def get_dataset_10crop_eval(val_filenames, val_labels):
    """ take in val filenames and labels,
        return batched standard 10 crop of one image.
    """
    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(val_preprocess_standard_10crop_eval,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types,
                                                       val_dataset.output_shapes)
    images_10crop, labels = iterator.get_next()

    val_data_init_op = iterator.make_initializer(val_dataset)

    return images_10crop, labels, val_data_init_op


def get_dataset_10crop_train_eval(train_filenames, train_labels, val_filenames, val_labels):
    raise NotImplementedError("input size not constant 224")
    """ take in train and validation filenames and labels,
        return batched standard 10 crop of one image.
    """
    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(val_preprocess_standard_10crop_eval,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(val_preprocess_standard_10crop_eval,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    assert val_dataset.output_types == train_dataset.output_types
    assert val_dataset.output_shapes == train_dataset.output_shapes
    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types,
                                                       val_dataset.output_shapes)
    images_10crop, labels = iterator.get_next()

    val_data_init_op = iterator.make_initializer(val_dataset)
    train_data_init_op = iterator.make_initializer(train_dataset)

    return images_10crop, labels, train_data_init_op, val_data_init_op


def get_dataset_30crop_eval(val_filenames, val_labels):
    """ take in val filenames and labels,
        return batched standard 30 crop of one image.
    """

    def val_preprocess_30crop_eval(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        height, width = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])

        enumerate_small_side = [c.input_size * 256 / 224]
        batched_image = [None for i in range(2)]
        side = c.input_size
        for i in range(1):
            smallest_side = tf.to_float(enumerate_small_side[i])
            scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
            centered_image = resized_image - means
            flipped_image = tf.image.flip_left_right(centered_image)

            batched_image[i * 2] = tf.cond(tf.greater(height, width),
                lambda: _crop_along_h(centered_image, new_height, new_width, side),
                lambda: _crop_along_w(centered_image, new_height, new_width, side))
            batched_image[i * 2 + 1] = tf.cond(tf.greater(height, width),
                lambda: _crop_along_h(flipped_image, new_height, new_width, side),
                lambda: _crop_along_w(flipped_image, new_height, new_width, side))

        return tf.concat(batched_image, 0), label

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(val_preprocess_30crop_eval,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types,
                                                       val_dataset.output_shapes)
    images_10crop, labels = iterator.get_next()

    val_data_init_op = iterator.make_initializer(val_dataset)

    return images_10crop, labels, val_data_init_op


def get_dataset_90crop_eval(val_filenames, val_labels):
    raise NotImplementedError("input size not constant 224")
    """ take in val filenames and labels,
        return batched standard 10 crop of one image.
    """
    def val_preprocess_90crop_eval(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        height, width = tf.to_float(tf.shape(image)[0]), tf.to_float(tf.shape(image)[1])

        enumerate_small_side = [228, 256, 284]
        batched_image = [None for i in range(6)]
        for i in range(3):
            smallest_side = tf.to_float(enumerate_small_side[i])
            scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
            centered_image = resized_image - means
            flipped_image = tf.image.flip_left_right(centered_image)

            batched_image[i * 2] = tf.cond(tf.greater(height, width),
                lambda: _crop_along_h(centered_image, new_height, new_width),
                lambda: _crop_along_w(centered_image, new_height, new_width))
            batched_image[i * 2 + 1] = tf.cond(tf.greater(height, width),
                lambda: _crop_along_h(flipped_image, new_height, new_width),
                lambda: _crop_along_w(flipped_image, new_height, new_width))

        return tf.concat(batched_image, 0), label

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(val_preprocess_90crop_eval,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types,
                                                       val_dataset.output_shapes)
    images_90crop, labels = iterator.get_next()

    val_data_init_op = iterator.make_initializer(val_dataset)

    return images_90crop, labels, val_data_init_op


def get_dataset_10crop_eval_with_filename(val_filenames, val_labels):
    """ take in val filenames and labels,
        return batched standard 10 crop of one image.
    """
    def val_preprocess_standard_10crop_eval_with_filename(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = input_size * 256 / 224.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
        centered_image = resized_image - means
        flipped_image = tf.image.flip_left_right(centered_image)

        return tf.concat([crop5(centered_image, new_height, new_width, input_size),
                          crop5(flipped_image, new_height, new_width, input_size)], 0), label, filename

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(val_preprocess_standard_10crop_eval_with_filename,
                                  num_threads=num_workers, output_buffer_size=num_workers)

    iterator = tf.contrib.data.Iterator.from_structure(val_dataset.output_types,
                                                       val_dataset.output_shapes)
    images_10crop, labels, filenames = iterator.get_next()

    val_data_init_op = iterator.make_initializer(val_dataset)

    return images_10crop, labels, filenames, val_data_init_op


def get_dataset_10crop_test(test_filenames):

    def test_preprocess_standard_10crop_test(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = input_size * 256 / 224.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        means = tf.reshape(tf.constant(IMAGENET_MEAN_RGB), [1, 1, 3])
        centered_image = resized_image - means
        flipped_image = tf.image.flip_left_right(centered_image)

        return tf.concat([crop5(centered_image, new_height, new_width, input_size),
                          crop5(flipped_image, new_height, new_width, input_size)], 0), filename

    # test dataset
    test_filenames = tf.constant(test_filenames)
    test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames))
    test_dataset = test_dataset.map(test_preprocess_standard_10crop_test,
        num_threads=num_workers, output_buffer_size=num_workers)

    iterator = tf.contrib.data.Iterator.from_structure(test_dataset.output_types,
                                                       test_dataset.output_shapes)
    images_10crop, filename= iterator.get_next()

    test_data_init_op = iterator.make_initializer(test_dataset)

    return images_10crop, filename, test_data_init_op


def main():
    raise NotImplementedError("input size not constant 224")
    train_filenames, train_labels = list_images('train')
    val_filenames, val_labels = list_images('validation')

    graph = tf.Graph()
    with graph.as_default():
        val_cropped_image, labels, data_init_op = \
            get_dataset_90crop_eval(val_filenames, val_labels)

    with tf.Session(graph=graph) as sess:
        sess.run(data_init_op)
        ima_dir = "/home/lyc/ai_challenger_scene/90crop_images"
        if not os.path.exists(ima_dir):
            os.makedirs(ima_dir)
        for _ in range(3):
            i, l = sess.run([val_cropped_image, labels])
            print("_: {}".format(_))
            print("type(i): {}".format(type(i)))
            print("i.shape: {}".format(i.shape))
            print("type(l): {}".format(type(l)))
            print("l.shape: {}".format(l.shape))
            # for y in range(10):
            # print("y[{}]: {}".format(y, i[y][0:100]))
            # im.save will automatically subtract min from matrix. Thus
            for j in range(90):
                filename = os.path.join(ima_dir, "{}-{}-{}.jpeg".format(_, l, j))
                scipy.misc.imsave(filename, i[j])


if __name__ == "__main__":
    main()
 