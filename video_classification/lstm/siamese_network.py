import tensorflow as tf
import numpy as np


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden = hidden_units
        n_layers = 1
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print x
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout, input_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout, input_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            # add an fully connection
            #outputs = tf.layers.dense(inputs=outputs[-1], units=embedding_size, activation=tf.nn.relu)
            initer = tf.truncated_normal_initializer(stddev=0.01)
            W = tf.get_variable("W", dtype=tf.float32, shape=[outputs[-1].get_shape()[1], embedding_size], initializer=initer)
            b = tf.get_variable('b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[embedding_size], dtype=tf.float32))
            outputs = tf.nn.bias_add(tf.matmul(outputs[-1], W), b)
        return outputs

    def contrastive_loss(self, y, d, reg_cost, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2 + reg_cost

    def __init__(
            self, sequence_length, embedding_size, hidden_units, l2_reg_lambda, batch_size, img_feat_dim):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length, img_feat_dim], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length, img_feat_dim], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope("output") as scope:
            self.out1 = self.BiRNN(self.input_x1, self.dropout_keep_prob, "side1", embedding_size,
                                   sequence_length, hidden_units)
            scope.reuse_variables()
            self.out2 = self.BiRNN(self.input_x2, self.dropout_keep_prob, "side1", embedding_size,
                                   sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            tv = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
            self.loss = self.contrastive_loss(self.input_y, self.distance, regularization_cost, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.TP = tf.count_nonzero(self.temp_sim * self.input_y)
            self.TN = tf.count_nonzero((self.temp_sim - 1) * (self.input_y - 1))
            self.FP = tf.count_nonzero(self.temp_sim * (self.input_y - 1))
            self.FN = tf.count_nonzero((self.temp_sim - 1) * self.input_y)
            self.precision = tf.divide(self.TP , self.TP + self.FP)
            self.recall = tf.divide(self.TP, self.TP + self.FN)
            self.f1 = 2 * self.precision * tf.divide(self.recall, self.precision + self.recall)
