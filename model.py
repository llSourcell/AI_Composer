import os
import logging
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq

import nottingham_util

class Model(object):
    """ 
    Cross-Entropy Naive Formulation
    A single time step may have multiple notes active, so a sigmoid cross entropy loss
    is used to match targets.

    seq_input: a [ T x B x D ] matrix, where T is the time steps in the batch, B is the
               batch size, and D is the amount of dimensions
    """
    
    def __init__(self, config, training=False):
        self.config = config
        self.time_batch_len = time_batch_len = config.time_batch_len
        self.input_dim = input_dim = config.input_dim
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout_prob = config.dropout_prob
        input_dropout_prob = config.input_dropout_prob
        cell_type = config.cell_type

        self.seq_input = \
            tf.placeholder(tf.float32, shape=[self.time_batch_len, None, input_dim])

        if (dropout_prob <= 0.0 or dropout_prob > 1.0):
            raise Exception("Invalid dropout probability: {}".format(dropout_prob))

        if (input_dropout_prob <= 0.0 or input_dropout_prob > 1.0):
            raise Exception("Invalid input dropout probability: {}".format(input_dropout_prob))

        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [hidden_size, input_dim])
            output_b = tf.get_variable("output_b", [input_dim])
            self.lr = tf.constant(config.learning_rate, name="learning_rate")
            self.lr_decay = tf.constant(config.learning_rate_decay, name="learning_rate_decay")

        def create_cell(input_size):
            if cell_type == "vanilla":
                cell_class = rnn_cell.BasicRNNCell
            elif cell_type == "gru":
                cell_class = rnn_cell.BasicGRUCell
            elif cell_type == "lstm":
                cell_class = rnn_cell.BasicLSTMCell
            else:
                raise Exception("Invalid cell type: {}".format(cell_type))

            cell = cell_class(hidden_size, input_size = input_size)
            if training:
                return rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_prob)
            else:
                return cell

        if training:
            self.seq_input_dropout = tf.nn.dropout(self.seq_input, keep_prob = input_dropout_prob)
        else:
            self.seq_input_dropout = self.seq_input

        self.cell = rnn_cell.MultiRNNCell(
            [create_cell(input_dim)] + [create_cell(hidden_size) for i in range(1, num_layers)])

        batch_size = tf.shape(self.seq_input_dropout)[0]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        inputs_list = tf.unpack(self.seq_input_dropout)

        # rnn outputs a list of [batch_size x H] outputs
        outputs_list, self.final_state = rnn.rnn(self.cell, inputs_list, 
                                                 initial_state=self.initial_state)

        outputs = tf.pack(outputs_list)
        outputs_concat = tf.reshape(outputs, [-1, hidden_size])
        logits_concat = tf.matmul(outputs_concat, output_W) + output_b
        logits = tf.reshape(logits_concat, [self.time_batch_len, -1, input_dim])

        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logits_concat)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay) \
                            .minimize(self.loss)

    def init_loss(self, outputs, _):
        self.seq_targets = \
            tf.placeholder(tf.float32, [self.time_batch_len, None, self.input_dim])

        batch_size = tf.shape(self.seq_input_dropout)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.seq_targets)
        return tf.reduce_sum(cross_ent) / self.time_batch_len / tf.to_float(batch_size)

    def calculate_probs(self, logits):
        return tf.sigmoid(logits)

    def get_cell_zero_state(self, session, batch_size):
        return self.cell.zero_state(batch_size, tf.float32).eval(session=session)

class NottinghamModel(Model):
    """ 
    Dual softmax formulation 

    A single time step should be a concatenation of two one-hot-encoding binary vectors.
    Loss function is a sum of two softmax loss functions over [:r] and [r:] respectively,
    where r is the number of melody classes
    """

    def init_loss(self, outputs, outputs_concat):
        self.seq_targets = \
            tf.placeholder(tf.int64, [self.time_batch_len, None, 2])
        batch_size = tf.shape(self.seq_targets)[1]

        with tf.variable_scope("rnnlstm"):
            self.melody_coeff = tf.constant(self.config.melody_coeff)

        r = nottingham_util.NOTTINGHAM_MELODY_RANGE
        targets_concat = tf.reshape(self.seq_targets, [-1, 2])

        melody_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            outputs_concat[:, :r], \
            targets_concat[:, 0])
        harmony_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            outputs_concat[:, r:], \
            targets_concat[:, 1])
        losses = tf.add(self.melody_coeff * melody_loss, (1 - self.melody_coeff) * harmony_loss)
        return tf.reduce_sum(losses) / self.time_batch_len / tf.to_float(batch_size)

    def calculate_probs(self, logits):
        steps = []
        for t in range(self.time_batch_len):
            melody_softmax = tf.nn.softmax(logits[t, :, :nottingham_util.NOTTINGHAM_MELODY_RANGE])
            harmony_softmax = tf.nn.softmax(logits[t, :, nottingham_util.NOTTINGHAM_MELODY_RANGE:])
            steps.append(tf.concat(1, [melody_softmax, harmony_softmax]))
        return tf.pack(steps)

    def assign_melody_coeff(self, session, melody_coeff):
        if melody_coeff < 0.0 or melody_coeff > 1.0:
            raise Exception("Invalid melody coeffecient")

        session.run(tf.assign(self.melody_coeff, melody_coeff))

class NottinghamSeparate(Model):
    """ 
    Single softmax formulation 
    
    Regular single classification formulation, used to train baseline models
    where the melody and harmony are trained separately
    """

    def init_loss(self, outputs, outputs_concat):
        self.seq_targets = \
            tf.placeholder(tf.int64, [self.time_batch_len, None])
        batch_size = tf.shape(self.seq_targets)[1]

        with tf.variable_scope("rnnlstm"):
            self.melody_coeff = tf.constant(self.config.melody_coeff)

        targets_concat = tf.reshape(self.seq_targets, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            outputs_concat, targets_concat)

        return tf.reduce_sum(losses) / self.time_batch_len / tf.to_float(batch_size)

    def calculate_probs(self, logits):
        steps = []
        for t in range(self.time_batch_len):
            softmax = tf.nn.softmax(logits[t, :, :])
            steps.append(softmax)
        return tf.pack(steps)
