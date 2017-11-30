#encoding=utf-8
import sys
import os
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq

import data_helpers


# hyper-parameters
num_epochs = 100
batch_size = 128
rnn_size = 256
rnn_layer_num = 2
embed_dim = 200
seq_length = 56
label_size = 2
learning_rate = 0.0001
# Show stats for every n number of batches
test_every_n_batches = 100
save_dir = './save'
pos_data_file = "./data/rt-polaritydata/rt-polarity.pos"
neg_data_file = "./data/rt-polaritydata/rt-polarity.neg"
test_sample_rate = 0.1


def get_inputs():
    
    input_ph = tf.placeholder(tf.int32, shape = (None, None), name = "input")
    label_ph = tf.placeholder(tf.int32, shape = (None, None), name = "label")
    #input_ph = tf.placeholder(tf.int32, shape = (batch_size, seq_length), name = "input")
    #label_ph = tf.placeholder(tf.int32, shape = (batch_size, label_size), name = "label")
    learning_rate_ph = tf.placeholder(tf.float64, name = "learning_rate")
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    true_len_list = tf.placeholder(tf.int32, shape = (None,), name = "true_len_list")

    return input_ph, label_ph, learning_rate_ph, dropout_keep_prob, true_len_list


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


def build_rnn(cell, inputs, true_len_list):

    outputs, state = tf.nn.dynamic_rnn(
        cell = cell,
        inputs = inputs,
        sequence_length = true_len_list,
        dtype = tf.float32)

    state = tf.identity(state, name = "final_state")

    return outputs, state


def build_nn(cell, rnn_size, rnn_layer_num, input_data, vocab_size, label_size, embed_dim, dp_prob, true_len_list):

    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed, true_len_list)
    # 取最后一个词的输出
    #outputs = tf.reshape(outputs[:,-1,:], [-1, rnn_size])
    outputs = tf.reshape(final_state[1][1], [-1, rnn_size])
    print "outputs shape:"
    print outputs.shape
    print "final state shape:"
    print final_state.shape

    #softmax_w = tf.get_variable(
    #    "softmax_w", [rnn_size, label_size], dtype = tf.float32)
    #softmax_b = tf.get_variable(
    #    "softmax_b", [label_size], dtype = tf.float32)
    softmax_w = tf.Variable(tf.random_normal([rnn_size, label_size]), name = "softmax_w", dtype = tf.float32)
    softmax_b = tf.Variable(tf.random_normal([label_size]), name = "softmax_b", dtype = tf.float32)
    logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    logits = tf.nn.dropout(logits, dp_prob)
    logits = tf.reshape(logits, [-1, label_size])
    print "logits shape:"
    print logits.shape

    return logits, final_state


def build_bidirectional_rnn(rnn_size, input_data, vocab_size, label_size, embed_dim, dp_prob, true_len_list):
    
    with tf.variable_scope(initializer=tf.orthogonal_initializer()):
        # get embedding layer
        embed = get_embed(input_data, vocab_size, embed_dim)
    
        # get cell
        lstm_forward = tf.contrib.rnn.LSTMBlockCell(rnn_size, forget_bias = 0.0)
        lstm_backward = tf.contrib.rnn.LSTMBlockCell(rnn_size, forget_bias = 0.0)
    
        # build bidirectional rnn
        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
            lstm_forward,
            lstm_backward,
            embed,
            sequence_length = true_len_list,
            dtype = tf.float32)

    final_state = tf.identity(final_state, name = "final_state")

    logits = tf.reshape(tf.concat([final_state[0][1], final_state[1][1]], 1), [-1, 2*rnn_size])
    print "forward outputs shape:"
    print outputs[0].shape
    print "backward outputs shape:"
    print outputs[1].shape
    print "final state shape:"
    print logits.shape

    #softmax_w = tf.get_variable(
    #    "softmax_w", [rnn_size, label_size], dtype = tf.float32)
    #softmax_b = tf.get_variable(
    #    "softmax_b", [label_size], dtype = tf.float32)
    softmax_w = tf.Variable(tf.random_normal([2*rnn_size, label_size]), name = "softmax_w", dtype = tf.float32)
    softmax_b = tf.Variable(tf.random_normal([label_size]), name = "softmax_b", dtype = tf.float32)
    logits = tf.nn.xw_plus_b(logits, softmax_w, softmax_b)
    logits = tf.nn.dropout(logits, dp_prob)
    logits = tf.reshape(logits, [-1, label_size])
    print "logits shape:"
    print logits.shape

    return logits, final_state


def get_vocabulary_and_input(x_text):

    # build vocabulary
    max_doc_len = max([len(x.split(" ")) for x in x_text])
    print "max doc len:%d" % max_doc_len
    temp_dict = {word:None for word in " ".join(x_text).split(" ")}
    key_list = temp_dict.keys()
    int_to_word = {idx+1:word for idx, word in enumerate(key_list)}
    word_to_int = {word:idx+1 for idx, word in enumerate(key_list)}
    x = []
    for sent in x_text:
        now_list = []
        word_list = sent.split(" ")
        for i in range(max_doc_len):
            if i < len(word_list):
                now_list.append(word_to_int[word_list[i]])
            else:
                now_list.append(0)
        x.append(now_list)
    x = np.array(x)
    return x, word_to_int, int_to_word



"""
Data preprocess
"""
# load data
x_text, y = data_helpers.load_data_and_labels(pos_data_file, neg_data_file)

# get vocabulary and input
x, word_to_int, int_to_word = get_vocabulary_and_input(x_text)

# Randomly shuffle data
np.random.seed(10)
shuffle_idx = np.random.permutation(len(y))
x_shuffled = x[shuffle_idx]
y_shuffled = y[shuffle_idx]

# Split train/test set
test_sample_num = int(test_sample_rate * len(y))
train_sample_num = len(y) - test_sample_num
x_train, x_test = x_shuffled[:train_sample_num], x_shuffled[train_sample_num:]
y_train, y_test = y_shuffled[:train_sample_num], y_shuffled[train_sample_num:]
print "train sample num:%d, test sample num:%d" % (train_sample_num, test_sample_num)

# get true length of every test sample
test_len_list = []
for i in range(y_test.shape[0]):
    now_data = x_test[i]
    idx = 0
    for j in range(now_data.shape[0]):
        if now_data[j] == 0:
            break
        idx += 1
    test_len_list.append(idx)


"""
Build training graph
"""
use_bidirectional_rnn = True
train_graph = tf.Graph()
with train_graph.as_default():
    #vocab_size = len(int_to_vocab)
    vocab_size = len(word_to_int)+1
    input_text, targets, lr, dp_prob, true_len_list= get_inputs()
    input_data_shape = tf.shape(input_text)
    if use_bidirectional_rnn:
        logits, final_state = build_bidirectional_rnn(rnn_size, input_text, vocab_size, label_size, embed_dim, dp_prob, true_len_list)
    else:
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, rnn_layer_num)
        print "init state shape:"
        print initial_state.shape
        logits, final_state = build_nn(cell, rnn_size, rnn_layer_num, input_text, vocab_size, label_size, embed_dim, dp_prob, true_len_list)
        print "final state shape:"
        print final_state.shape
        print "logits shape:"
        print logits.shape

    # Probabilities for generating words
    #probs = tf.nn.softmax(logits, name='probs')
    probs = logits
    print "probs shape:"
    print probs.shape

    # Loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=probs, labels=targets))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, tf.train.get_or_create_global_step())

    correct_pred = tf.equal(tf.argmax(probs,1), tf.argmax(targets,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




"""
Train
"""
with tf.Session(graph = train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch, tll, batch_num):

        feed_dict = {
            input_text: x_batch,
            targets: y_batch,
            dp_prob: 0.5,
            lr: learning_rate,
            true_len_list: tll
        }

        _, loss, acc = sess.run(
            [train_op, cost, accuracy], feed_dict)
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%H:%M:%S")
        if batch_num != 0 and batch_num % 10 == 0:
            print "%s   loss: %.3f, acc: %.3f" % (now_time, loss, acc)

    def test_step(x_batch, y_batch, tll):

        feed_dict = {
            input_text: x_batch,
            targets: y_batch,
            dp_prob: 1.0,
            true_len_list: tll
        }

        loss, acc = sess.run(
            [cost, accuracy], feed_dict)
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%H:%M:%S")
        print "\nEvaluation:"
        print "%s   loss: %.3f, acc: %.3f\n" % (now_time, loss, acc)


    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), batch_size, num_epochs)

    # Training loop
    steps = 0
    for batch, tll in batches:
        if batch.shape[0] != batch_size:
            continue
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, tll, steps)
        if steps != 0 and steps % test_every_n_batches == 0:
            test_step(x_test, y_test, test_len_list)
        steps += 1
