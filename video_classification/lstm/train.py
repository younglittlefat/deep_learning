#! /usr/bin/env python

import re
import os
import time
import datetime
import gc
from random import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from siamese_network import SiameseLSTM
from input_helper import InputHelper

# Parameters
# ==================================================

root_dir = "/home/younglittlefat/deep_learning/video_classification"

tf.app.flags.DEFINE_integer("embedding_dim", 384, "Dimensionality of character embedding (default: 300)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 1.0)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.app.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")  #for sentence semantic similarity use "train_snli.txt"
tf.app.flags.DEFINE_integer("hidden_units", 512, "Number of hidden units (default:50)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("img_feat_dim", 1024, "Feature dimension of images (default: 1024)")
tf.app.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.app.flags.FLAGS
import sys
FLAGS(sys.argv)

#FLAGS._parse_flags()
print "\nParameters:"
for attr, value in sorted(FLAGS.__flags.items()):
    print "%s=%s" % (attr.upper(), value)
print ""

#if FLAGS.training_files==None:
#    print "Input Files List is empty. use --training_files argument."
#    exit()


max_document_length=30
ih = InputHelper()
ih.load_data(os.path.join(root_dir, "video_features"), os.path.join(root_dir, "label/label"))

# Training
# ==================================================
print "starting graph def"
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
        learning_rate = tf.train.exponential_decay(0.001, global_step, decay_steps = 30, decay_rate = 0.9)
        lr_summary = tf.summary.scalar("learning_rate", learning_rate)
        #learning_rate = 0.002
        optimizer = tf.train.AdamOptimizer(learning_rate)
        print "initialized siameseModel object"
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print "defined training_ops"
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            print g
            grad_hist_summary = tf.summary.histogram("%s/grad/hist" % (v.name), g)
            sparsity_summary = tf.summary.scalar("%s/grad/sparsity" % (v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print "defined gradient summaries"
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print "Writing to %s\n" % (out_dir)

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)
    precision_summary = tf.summary.scalar("precision", siameseModel.precision)
    recall_summary = tf.summary.scalar("recall", siameseModel.recall)
    f1_summary = tf.summary.scalar("F1", siameseModel.f1)

    # Train Summaries
    train_summary_op = tf.summary.merge([lr_summary, loss_summary, acc_summary, grad_summaries_merged, precision_summary, recall_summary, f1_summary])
    #train_summary_op = tf.summary.merge([lr_summary, loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, recall_summary, f1_summary])
    #dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    #vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random() >= 0:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist, sim, summaries, f1 = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim, train_summary_op, siameseModel.f1],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print "TRAIN %s: step %s, loss %f, acc %f, f1 %f" % (time_str, step, loss, accuracy, f1)
        train_summary_writer.add_summary(summaries, step)
        #print y_batch, dist, sim

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """ 
        if random() >= 0:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim, summaries, f1 = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim, dev_summary_op, siameseModel.f1],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print "DEV %s: step %s, loss %f, acc %f, f1 %f" % (time_str, step, loss, accuracy, f1)
        dev_summary_writer.add_summary(summaries, step)
        #print y_batch, sim
        return accuracy

    # Generate batches
    batches = ih.batch_iter(0, FLAGS.batch_size, FLAGS.num_epochs)

    ptr=0
    max_validation_acc=0.0
    sum_no_of_batches = int(ih.train_set[2].shape[0] / FLAGS.batch_size) + 1
    print "sum_no_of_batches: %d" % (sum_no_of_batches)
    for nn in xrange(sum_no_of_batches*FLAGS.num_epochs):
        batch = batches.next()
        if len(batch)<1:
            continue
        x1_batch,x2_batch, y_batch = zip(*batch)
        if len(y_batch)<1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc=0.0
        if current_step % FLAGS.evaluate_every == 0:
            print "\nEvaluation:"
            dev_batches = ih.batch_iter(1, ih.dev_set[2].shape[0], 1)
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print "Saved model %s with sum_accuracy=%f checkpoint to %s}\n" % (nn, max_validation_acc, checkpoint_prefix)
