import os
import sys

import numpy as np
import tensorflow as tf

from siamese_network import SiameseLSTM
from input_helper_for_predict import InputHelper

root_dir = "/home/younglittlefat/deep_learning/video_classification"
tf.app.flags.DEFINE_integer("embedding_dim", 384, "Dimensionality of character embedding (default: 300)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 1.0)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.app.flags.DEFINE_integer("hidden_units", 512, "Number of hidden units (default:50)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("img_feat_dim", 1024, "Feature dimension of images (default: 1024)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.app.flags.FLAGS
import sys
FLAGS(sys.argv)

max_document_length=30
ih = InputHelper()
ih.load_data(os.path.join(root_dir, "video_features"), os.path.join(root_dir, "label/label_for_embedding"))
batches = ih.batch_iter(0, FLAGS.batch_size, 1)
model_path = sys.argv[1]

tf.reset_default_graph()

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print "started session"
    with sess.as_default():
        siameseModel = SiameseLSTM(
            sequence_length=max_document_length,
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            img_feat_dim = FLAGS.img_feat_dim
        )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        sum_no_of_batches = int(ih.train_set[2].shape[0] / FLAGS.batch_size) + 1
        # begin to predict
        for nn in xrange(sum_no_of_batches):
            batch = batches.next()
            x1, x2, y = zip(*batch)
            feed_dict = {
                siameseModel.input_x1: x1,
                siameseModel.input_x2: x2,
                siameseModel.input_y: y,
                siameseModel.dropout_keep_prob: 1.0,
            }
            out = sess.run([siameseModel.out1], feed_dict)
            for i in range(len(ih.name_list)):
                print "%s %s" % (ih.name_list[i][0], " ".join(map(str, out[0][i].tolist())))
