"""Create the input data pipeline using `tf.data`"""
import os

import tensorflow as tf
import numpy as np

import model.mnist_dataset as mnist_dataset


def train_input_fn(data_dir, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.train(data_dir)
    dataset = dataset.shuffle(params.train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    print dataset
    return dataset


def test_input_fn(data_dir, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.test(data_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def my_train_input_fn(data_dir, params):
    """

    """

    def _parse_function(example_proto):
        features = {
            'label': tf.FixedLenFeature([],tf.int64),
            #'data': tf.FixedLenFeature([],tf.float32)
            'data': tf.FixedLenFeature([],tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.decode_raw(parsed_features["data"], tf.float32)
        data = tf.reshape(data, [30, 1024])
        return data, parsed_features["label"]

    label_path = params.label_path
    file_list = []
    with open(label_path, "r") as f:
        for line in f:
            temp = line.strip().split(" ")
            tf_file = temp[0].split(".")[0] + ".tfrecords"
            file_list.append(os.path.join(data_dir, os.path.join("tfrecords", tf_file)))

    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(params.train_size).repeat(params.num_epochs).batch(params.batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

