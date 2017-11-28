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


_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    input_tensor = loaded_graph.get_tensor_by_name("input:0")
    print "input tensor:"
    print input_tensor.shape
    init_state_tensor = loaded_graph.get_tensor_by_name("initial_state:0")
    print "init state tensor:"
    print init_state_tensor.shape
    final_state_tensor = loaded_graph.get_tensor_by_name("final_state:0")
    print "final state tensor:"
    print final_state_tensor.shape
    probs_tensor = loaded_graph.get_tensor_by_name("probs:0")
    print "probs tensor:"
    print probs_tensor.shape

    return (input_tensor, init_state_tensor, final_state_tensor, probs_tensor)


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """

    return int_to_vocab[np.argmax(probabilities)]


"""
Test
"""
#tests.test_get_tensors(get_tensors)


"""
Generate
"""
load_dir = "./save"
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':', "what's", "the", "matter", "||comma||", \
        "homer", "||quotation_mark||", "the", "depressin'", "effects", "of", \
        "alcohol", "usually", "don't", "kick"]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
    print "\nprev state shape:"
    print prev_state.shape
    print "seq length:"
    print seq_length

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)
        gen_sentences.append(pred_word)

    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')

    print tv_script
