#encoding=utf-8
import sys
import os
import json
import datetime

import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion
import warnings

import helper
import problem_unittests as tests

data_dir = "./data/simpsons/moes_tavern_lines.txt"
raw_text = helper.load_data(data_dir)
raw_text = raw_text[81:]


def explore_data(raw_text):
    view_sentence_range = (0, 10)
    print "Dataset Stat"
    print "Number of unique words: %d" % (len({word: None for word in raw_text.split()}))
    scenes = raw_text.split('\n\n')
    print "Number of scenes: %d" % len(scenes)
    sentence_count_scene = [scene.count('\n') for scene in scenes]
    print "Average number of sentences in each scene: %d" % np.average(sentence_count_scene)
    sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
    print "Number of lines: %d" % len(sentences)
    word_count_sentence = [len(sentence.split()) for sentence in sentences]
    print "Average number of words in each line: %d\n" % np.average(word_count_sentence)
    print "The sentences %d to %d:" % (view_sentence_range[0], view_sentence_range[1])
    print "\n".join(raw_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]])


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_list = {word: None for word in text}.keys()
    vocab_to_int = {word: idx for idx, word in enumerate(vocab_list)}
    int_to_vocab = {idx: word for idx, word in enumerate(vocab_list)}

    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    punc_token_mapping = {}
    punc_token_mapping["."] = "||Period||"
    punc_token_mapping[","] = "||Comma||"
    punc_token_mapping["\""] = "||Quotation_Mark||"
    punc_token_mapping["!"] = "||Exclamation_Mark||"
    punc_token_mapping["?"] = "||Question_Mark||"
    punc_token_mapping[";"] = "||Semicolon||"
    punc_token_mapping["("] = "||Left_Parentheses||"
    punc_token_mapping[")"] = "||Right_Parentheses||"
    punc_token_mapping["--"] = "||Dash||"
    punc_token_mapping["\n"] = "||Return||"

    return punc_token_mapping


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input_ph = tf.placeholder(tf.int32, shape = (None, None), name = "input")
    targets_ph = tf.placeholder(tf.int32, shape = (None, None), name = "targets")
    learning_rate_ph = tf.placeholder(tf.float64, name = "learning_rate")

    return (input_ph, targets_ph, learning_rate_ph)


def _get_lstm_cell(rnn_size):
    rnn_mode = "BLOCK"
    if rnn_mode == "BASIC":
        return tf.contrib.rnn.BasicLSTMCell(
            rnn_size, forget_bias = 0.0, state_is_tuple = True,
            reuse = False)
    if rnn_mode == "BLOCK":
        return tf.contrib.rnn.LSTMBlockCell(
            rnn_size, forget_bias = 0.0)


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    cell = _get_lstm_cell(rnn_size)

    cell = tf.contrib.rnn.MultiRNNCell(
        [cell for _ in range(rnn_size/128)], state_is_tuple = True)

    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name = "initial_state")

    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """

    embedding = tf.get_variable(
        "embedding", [vocab_size, embed_dim], dtype = tf.float32)
    return tf.nn.embedding_lookup(embedding, input_data)


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, state = tf.nn.dynamic_rnn(
        cell = cell,
        inputs = inputs,
        dtype = tf.float32)

    state = tf.identity(state, name = "final_state")

    return (outputs, state)


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """

    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)
    outputs = tf.reshape(outputs, [-1, rnn_size])

    softmax_w = tf.get_variable(
        "softmax_w", [rnn_size, vocab_size], dtype = tf.float32)
    softmax_b = tf.get_variable(
        "softmax_b", [vocab_size], dtype = tf.float32)
    logits = tf.nn.xw_plus_b(outputs, softmax_w, softmax_b)
    logits = tf.reshape(logits, [-1, seq_length, vocab_size])

    return (logits, final_state)


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """

    n_batches = int(len(int_text) / (batch_size * seq_length))
    # Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    ydata[-1] = xdata[0]
    
    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)
    return np.array(list(zip(x_batches, y_batches)))


"""
Test part
"""
#explore_data(raw_text)
#tests.test_create_lookup_tables(create_lookup_tables)
#tests.test_tokenize(token_lookup)
#tests.test_get_inputs(get_inputs)
#tests.test_get_init_cell(get_init_cell)
#tests.test_get_embed(get_embed)
#tests.test_build_rnn(build_rnn)
#tests.test_build_nn(build_nn)
#tests.test_get_batches(get_batches)
#helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


"""
Main part
"""
int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

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

"""
build training graph
"""
from tensorflow.contrib import seq2seq
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    print "init state shape:"
    print initial_state.shape
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
    print "final state shape:"
    print final_state.shape
    print "logits shape:"
    print logits.shape

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')
    print "probs shape:"
    print probs.shape

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

"""
train
"""
batches = get_batches(int_text, batch_size, seq_length)
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                #print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                now_time = datetime.datetime.now()
                now_time = now_time.strftime("%H:%M:%S")
                print '%s  Epoch %d Batch %d/%d   train_loss = %.3f' % (now_time, epoch_i, batch_i, len(batches), train_loss)

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print 'Model Trained and Saved'

# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))

