""" scene data reader """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc

import scene_model

FLAGS = None

NUM_CLASSES = 80
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 53879
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 7120

TRAIN_TFRECORD_FILE = 'train.tfrecords'
VALIDATION_TFRECORD_FILE = "validation.tfrecords"


def read_and_decode(filename_queue):
    """ read from tf record, return original image without
        distort or reshape to standard shape.
    :param filename_queue: tensorflow file name queue
    :return: image, label
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size, num_epochs):
    """Construct distorted input for scene training using the Reader ops.

    Args:
        data_dir: Path to the scene data directory.
        batch_size: Number of images per batch.
        num_epochs: Number of epochs

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, TRAIN_TFRECORD_FILE)]

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs)

    # Read examples from files in the filename queue.
    image, label = read_and_decode(filename_queue)
    image = tf.cast(image, tf.float32)

    input_size = tf.constant((scene_model.INPUT_HEIGHT, scene_model.INPUT_WIDTH), dtype=tf.int32)

    resized_image = tf.image.resize_images(image, input_size)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(resized_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size, num_epochs):
    """Construct input for evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the scene data directory.
        batch_size: Number of images per batch.
        num_epochs: Number of epochs

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, TRAIN_TFRECORD_FILE)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, VALIDATION_TFRECORD_FILE)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs)

    # Read examples from files in the filename queue.
    image, label = read_and_decode(filename_queue)
    image = tf.cast(image, tf.float32)

    input_size = tf.constant((scene_model.INPUT_HEIGHT, scene_model.INPUT_WIDTH), dtype=tf.int32)

    resized_image = tf.image.resize_images(image, input_size)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(resized_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def experiment():
    with tf.name_scope('input'):
        image, label = inputs(True, FLAGS.data_dir, 1, FLAGS.num_epochs)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                output_label, output_image = sess.run([label, image])
                if step % 100 == 0:
                    print('step: {}'.format(step))
                # print("type(label): {}".format(type(output_label)))
                # print("label.shape: {}".format(output_label.shape))
                # print("type(image): {}".format(type(output_image)))
                # print("image.shape: {}".format(output_image.shape))
                # print("type(output_image): {}".format(output_image))
                output_image = np.reshape(output_image, [224, 224, 3])
                save_image_filename = os.path.join("/scratch/lyc/tmp",
                                                   "image{}.jpeg".format(step))
                scipy.misc.toimage(output_image, cmin=0, cmax=255).save(save_image_filename)
                # im = Image.fromarray(output_image)
                # im.save(save_image_filename)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)


def main(unused_argv):
    experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        help='number of epochs.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/scratch/lyc/ai_challenger_scene/data',
        help='Directory to data files.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)