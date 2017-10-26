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

data_dir = "/scratch/lyc/ai_challenger_scene/data"

json_train = "ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json"
image_file_train = "ai_challenger_scene_train_20170904/scene_train_images_20170904"
json_validation = "ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json"
image_file_validation = "ai_challenger_scene_validation_20170908/scene_validation_images_20170908"

num_workers = 32

vgg_mean = [123.68, 116.78, 103.94]

rotate_angle = 45

num_classes = 80
num_examples_per_epoch_for_train = 53879
num_examples_per_epoch_for_val = 7120


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


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


def _parse_function_with_random_scale(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)

    # get random side from [256, 480]
    smallest_side = 256 + (480 - 256) * random_ops.random_uniform([1], 0, 1)[0]
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label


def _random_rotate(image, angle):
    """
    random rotate image. [-angle, angle] uniform distribution.
    :param image: image
    :param angle: range of rotate angle. degree.
    :return: rotated image tensor
    """
    angle = angle * math.pi / float(180)
    # rotate_factor = random_ops.random_uniform([1], -1, 1)[0]
    rotate_factor = random_ops.truncated_normal([1], mean=0.0, stddev=0.5)[0]
    random_angle = tf.multiply(rotate_factor, angle)
    return image_ops.rotate(image, random_angle, interpolation="BILINEAR")


def _apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                   for case in range(num_cases)])[0]


def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        return tf.clip_by_value(image, 0.0, 1.0)


def _color_perturb_slow(image_tensor):
    img_float_zero_one_range = tf.to_float(image_tensor) / 255
    distorted_image = _apply_with_random_selector(img_float_zero_one_range,
                                                  lambda x, ordering: _distort_color(x, ordering,
                                                                                     fast_mode=False),
                                                  num_cases=4)
    img_float_distorted_original_range = distorted_image * 255
    return img_float_distorted_original_range


def get_dataset(train_filenames, train_labels, val_filenames, val_labels, batch_size):
    def training_preprocess(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = flip_image - means

        return centered_image, label

    def val_preprocess(image, label):
        # resize_image_with_crop_or_pad will leave out some margin.???
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = crop_image - means

        return centered_image, label

    def val_preprocess_experiment(image, label):
        crop_image = tf.image.resize_images(image, [224, 224])

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = crop_image - means

        return centered_image, label

    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.map(training_preprocess,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(batch_size)

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
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


def get_dataset_with_rotate(train_filenames, train_labels, val_filenames, val_labels, batch_size):
    def training_preprocess(image, label):
        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = image - means

        rotated_image = _random_rotate(centered_image, rotate_angle)

        crop_image = tf.random_crop(rotated_image, [224, 224, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)

        return flip_image, label

    def val_preprocess(image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = crop_image - means
        return centered_image, label

    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.map(training_preprocess,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(batch_size)

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
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


def get_dataset_with_random_scale(train_filenames, train_labels, val_filenames, val_labels, batch_size):
    def training_preprocess(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = flip_image - means

        return centered_image, label

    def val_preprocess(image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = crop_image - means

        return centered_image, label

    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function_with_random_scale,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.map(training_preprocess,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(batch_size)

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
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


def get_dataset_with_color_augmentation(train_filenames, train_labels, val_filenames, val_labels, batch_size):
    def training_preprocess(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])

        color_perturb_image = _color_perturb_slow(crop_image)

        means = tf.constant(vgg_mean, dtype=tf.float32)
        means = tf.reshape(means, [1, 1, 3])
        centered_image = tf.subtract(color_perturb_image, means)

        flip_image = tf.image.random_flip_left_right(centered_image)
        return flip_image, label

    def val_preprocess(image, label):
        crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = crop_image - means

        return centered_image, label

    # Training dataset
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.map(training_preprocess,
                                      num_threads=num_workers, output_buffer_size=batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(batch_size)

    # Validation dataset
    val_filenames = tf.constant(val_filenames)
    val_labels = tf.constant(val_labels)
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


def get_dataset_10crop_eval(val_filenames, val_labels):
    """ take in val filenames and labels,
        return batched standard 10 crop of one image.
    """
    def _crop5(image, height, width):
        batched_image = tf.reshape(image, [-1, height, width, 3])

        def _crop(offset):
            patch = tf.image.extract_glimpse(batched_image, [224, 224],
                                             offsets=tf.to_float([offset]),
                                             centered=False, normalized=False)
            return tf.reshape(patch, [224, 224, 3])

        half_height, half_width = tf.to_int32(height / 2), tf.to_int32(width / 2)
        mid = _crop([half_height, half_width])
        l_u = _crop([112, 112])
        l_d = _crop([height - 112, 112])
        r_u = _crop([112, width - 112])
        r_d = _crop([height - 112, width - 112])

        return tf.stack([mid, l_u, l_d, r_u, r_d])

    def val_preprocess_standard_10crop_eval(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
        image = tf.cast(image_decoded, tf.float32)

        smallest_side = 256.0
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        height = tf.to_float(height)
        width = tf.to_float(width)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(height * scale)
        new_width = tf.to_int32(width * scale)

        resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = resized_image - means
        flipped_image = tf.image.flip_left_right(centered_image)

        return tf.concat([_crop5(centered_image, new_height, new_width),
                          _crop5(flipped_image, new_height, new_width)], 0), label

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


def get_dataset_random_rotate_experiment(train_filenames, train_labels):
    def image_preprocess_experiment(image, label):
        print("type(image): {}".format(type(image)))
        print("image.shape: {}".format(image.shape))
        print("vgg_mean: {}".format(vgg_mean))
        means = tf.constant(vgg_mean, dtype=tf.float32)
        means = tf.reshape(means, [1, 1, 3])
        centered_image = tf.subtract(image, means)

        rotated_image = _random_rotate(centered_image, 30)
        print("type(rotated_image): {}".format(type(rotated_image)))
        print("rotatd_image.shape: {}".format(rotated_image.shape))

        crop_image = tf.random_crop(rotated_image, [224, 224, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)
        print("filp_image.shape: {}".format(flip_image.shape))
        return flip_image, label

    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function, num_threads=1,
                                      output_buffer_size=10)
    train_dataset = train_dataset.map(image_preprocess_experiment, num_threads=1,
                                      output_buffer_size=10)

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
    images, labels = iterator.get_next()

    train_data_init_op = iterator.make_initializer(train_dataset)

    return images, labels, train_data_init_op


def get_dataset_random_scale_experiment(train_filenames, train_labels):
    def image_preprocess_experiment(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])
        flip_image = tf.image.random_flip_left_right(crop_image)

        means = tf.reshape(tf.constant(vgg_mean), [1, 1, 3])
        centered_image = flip_image - means

        return centered_image, label

    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function_with_random_scale, num_threads=1,
                                      output_buffer_size=10)
    train_dataset = train_dataset.map(image_preprocess_experiment, num_threads=1,
                                      output_buffer_size=10)

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
    images, labels = iterator.get_next()

    train_data_init_op = iterator.make_initializer(train_dataset)

    return images, labels, train_data_init_op


def get_dataset_color_aug_experiment(train_filenames, train_labels):
    def image_preprocess_experiment(image, label):
        crop_image = tf.random_crop(image, [224, 224, 3])

        color_perturb_image = _color_perturb_slow(crop_image)

        means = tf.constant(vgg_mean, dtype=tf.float32)
        means = tf.reshape(means, [1, 1, 3])
        centered_image = tf.subtract(color_perturb_image, means)

        flip_image = tf.image.random_flip_left_right(centered_image)
        return flip_image, label

    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function, num_threads=1,
                                      output_buffer_size=10)
    train_dataset = train_dataset.map(image_preprocess_experiment, num_threads=1,
                                      output_buffer_size=10)

    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
    images, labels = iterator.get_next()

    train_data_init_op = iterator.make_initializer(train_dataset)

    return images, labels, train_data_init_op


def main1():
    train_filenames, train_labels = list_images('train')
    val_filenames, val_labels = list_images('validation')

    graph = tf.Graph()
    with graph.as_default():
        val_cropped_image, labels, data_init_op = \
            get_dataset_10crop_eval(val_filenames, val_labels)

    with tf.Session(graph=graph) as sess:
        sess.run(data_init_op)
        ima_dir = "/scratch/lyc/ai_challenger_scene/std_10crop_images"
        if not os.path.exists(ima_dir):
            os.makedirs(ima_dir)
        for _ in range(50):
            i, l = sess.run([val_cropped_image, labels])
            print("_: {}".format(_))
            print("type(i): {}".format(type(i)))
            print("i.shape: {}".format(i.shape))
            print("type(l): {}".format(type(l)))
            print("l.shape: {}".format(l.shape))
            # for y in range(10):
            # print("y[{}]: {}".format(y, i[y][0:100]))
            # im.save will automatically subtract min from matrix. Thus
            for j in range(1):
                filename = os.path.join(ima_dir, "{}-{}-{}.jpeg".format(_, l, j))
                scipy.misc.imsave(filename, i[j])


def main2():
    train_filenames, train_labels = list_images('train')
    val_filenames, val_labels = list_images('validation')

    graph = tf.Graph()
    with graph.as_default():
        image, labels, train_init_op, val_init_op = \
            get_dataset(train_filenames, train_labels, val_filenames, val_labels, 32)

    with tf.Session(graph=graph) as sess:
        sess.run(val_init_op)
        ima_dir = "/scratch/lyc/ai_challenger_scene/std_val_images"
        if not os.path.exists(ima_dir):
            os.makedirs(ima_dir)
        for _ in range(2):
            i, l = sess.run([image, labels])
            print("_: {}".format(_))
            print("type(i): {}".format(type(i)))
            print("i.shape: {}".format(i.shape))
            print("type(l): {}".format(type(l)))
            print("l.shape: {}".format(l.shape))
            # for y in range(10):
            # print("y[{}]: {}".format(y, i[y][0:100]))
            # im.save will automatically subtract min from matrix. Thus
            for j in range(32):
                filename = os.path.join(ima_dir, "{}-{}-{}.jpeg".format(_, l[j], j))
                scipy.misc.imsave(filename, i[j])


if __name__ == "__main__":
    main1()
    main2()
