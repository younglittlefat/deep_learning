#encoding=utf-8
import sys
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq

import data_helper


# hyper-parameters
num_epochs = 10
batch_size = 128
rnn_size = 256
embed_dim = 300
seq_length = 15
learning_rate = 0.005
# Show stats for every n number of batches
show_every_n_batches = 1
save_dir = './save'


def get_inputs():
    
    input_ph = tf.placeholder(tf.int32, shape = (None, None), name = "input")
    label_ph = tf.placeholder(tf.int32, shape = (None, None), name = "label")
    learning_rate_ph = tf.placeholder(tf.float64, name = "learning_rate")

    return input_ph, label_ph, learning_rate_ph


def get_init_cell(batch_size, rnn_size, rnn_layer_num):

    cell = tf.contrib.rnn.LSTMBlockCell(
        rnn_size, forget_bias = 0.0)

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(rnn_layer_num)], state_is_tuple = True)

    init_state = cell.zero_state(batch_size, tf.float32)
    init_state = tf.identity(init_state, name = "initial_state")

    return cell, init_state


def get_embed(input_data, vocab_size, embed_dim):

    embed = tf.get_variable(
        "embedding", [vocab_size, embed_dim], dtype = tf.float32)
    return tf.nn.embedding_lookup(embed, input_data)


def build_rnn(cell, inputs):

    outputs, state = tf.nn.dynamic_rnn(
        cell = cell,
        inputs = inputs,
        dtype = tf.float32)

    state = tf.identity(state, name = "final_state")

    return outputs, state


def build_nn(cell, rnn_size, rnn_layer_num, input_data, vocab_size, embed_dim):

    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)
    outputs = tf.reshape(outputs, [-1, rnn_size])

    softmax_w = tf.getvariable(
        "softmax_w", [rnn_size, vocab_size], dtype = tf.float32)
    softmax_b = tf.get_variable(
        "softmax_b", [vocab_size], dtype = tf.float32)
    logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    logits = tf.reshape(logits, [-1, seq_length, vocab_size])

    return logits, final_state






