""" convert data to tf record
    do not convert to tf record. too large.
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
from PIL import Image

FLAGS = None

JSON_FILE_TRAIN = "ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json"
IMAGE_FILE_TRAIN = "ai_challenger_scene_train_20170904/scene_train_images_20170904"
JSON_FILE_VALID = "ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json"
IMAGE_FILE_VALID = "ai_challenger_scene_validation_20170908/scene_validation_images_20170908"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_tf_record(data_set_type):
    if data_set_type == "train":
        json_file_path = os.path.join(FLAGS.data_dir, JSON_FILE_TRAIN)
        image_dir = os.path.join(FLAGS.data_dir, IMAGE_FILE_TRAIN)
    elif data_set_type == "validation":
        json_file_path = os.path.join(FLAGS.data_dir, JSON_FILE_VALID)
        image_dir = os.path.join(FLAGS.data_dir, IMAGE_FILE_VALID)
    else:
        raise ValueError("Wrong data set type.")

    print("reading json file from: {}".format(json_file_path))
    with open(json_file_path, "r") as f:
        for line in f:
            image_label = json.loads(line)
    image_number = len(image_label)
    print("image number: {}".format(image_number))

    record_filename = os.path.join(FLAGS.data_dir, data_set_type + ".tfrecords")
    print('writing', record_filename)
    writer = tf.python_io.TFRecordWriter(record_filename)
    for i in range(image_number):
        if i % 100 == 0:
            print(i)
        image_name = image_label[i]["image_id"]
        image_raw = Image.open(os.path.join(image_dir, image_name))
        image_raw = np.asarray(image_raw, np.uint8)
        shape = np.asarray(image_raw.shape, np.int32)
        height, width = shape[0], shape[1]
        label = image_label[i]["label_id"]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(image_raw.tobytes())}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    convert_tf_record("train")
    convert_tf_record("validation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/scratch/lyc/ai_challenger_scene/data',
        help='Directory to download data files and write the converted result'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
