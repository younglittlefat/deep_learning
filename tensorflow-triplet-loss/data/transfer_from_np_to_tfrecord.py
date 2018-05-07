import sys
import os
import codecs

import tensorflow as tf
import numpy as np

tfrecord_dir = "./tfrecords"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(filename, filepath, label):
    filename = filename.split(".")[0] + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir, filename))
    #with open(filepath, "rb") as f:
    #    data_arr = np.load(f)
    data_arr = np.load(filepath)
    data_arr = data_arr.reshape(-1)
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(int(label)),
        'data': _bytes_feature(data_arr.tobytes())}))
        #'data': _float_feature(data_arr.tolist())}))
    writer.write(example.SerializeToString())
    writer.close()

def convert_all_features(label_file):
    data_dir = "/home/younglittlefat/deep_learning/video_classification/video_features"
    with open(label_file, "r") as f:
        for line in f:
            if line.startswith(codecs.BOM_UTF8):
                line = line[len(codecs.BOM_UTF8):]
            filename, label = line.strip().split(" ")
            label = int(label)
            filename = filename.split(".")[0] + ".npy"
            filepath = os.path.join(data_dir, filename)
            convert(filename, filepath, label)


def check_tfrecord(path):

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
        

    filenames = [path]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        print sess.run(one_element)


if __name__ == "__main__":
    #convert("20171023_18047378.npy", "/home/younglittlefat/deep_learning/video_classification/video_features/20171023_18047378.npy", 25)
    #convert_all_features("label")
    check_tfrecord("./tfrecords/20171023_18047378.tfrecords")
