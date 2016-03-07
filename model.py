import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq

import nottingham_util

class Model(object):
    """ RNN Model """
    
    def __init__(self, config, training=False):

        self.batch_size = batch_size = config["batch_size"]
        self.time_batch_len = time_batch_len = config["time_batch_len"]
        self.input_dim = input_dim = config["input_dim"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        dropout_prob = config["dropout_prob"]
        cell_type = config["cell_type"]

        print config

        if (dropout_prob <= 0.0 or dropout_prob > 1.0):
            raise Exception("Invalid dropout probability: {}".format(dropout_prob))

        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [hidden_size, input_dim])
            output_b = tf.get_variable("output_b", [input_dim])
            self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)
            self.lr_decay = tf.Variable(0.0, name="learning_rate_decay", trainable=False)

        if cell_type == "vanilla":
            first_layer = rnn_cell.BasicRNNCell(hidden_size, input_size = input_dim)  
            hidden_layer = rnn_cell.BasicRNNCell(hidden_size, input_size = hidden_size)
        else:
            first_layer = rnn_cell.BasicLSTMCell(hidden_size, input_size = input_dim)  
            hidden_layer = rnn_cell.BasicLSTMCell(hidden_size, input_size = hidden_size)

        cell = rnn_cell.MultiRNNCell([first_layer] + [hidden_layer] * (1-num_layers))
        if training and dropout_prob < 1.0:
            cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_prob)

        self.seq_input = \
            tf.placeholder(tf.float32, [time_batch_len, batch_size, input_dim])
        self.seq_input_lengths = \
            tf.placeholder(tf.int32, [batch_size])
        self.seq_targets = \
            tf.placeholder(tf.float32, [time_batch_len, batch_size, input_dim])
        self.unrolled_lengths = \
            tf.placeholder(tf.float32, [batch_size])

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # make inputs/targets a list of [batch_size x D] time-steps
        inputs = [tf.reshape(i, (batch_size, input_dim)) for i in 
                  tf.split(0, time_batch_len, self.seq_input)]
        targets = [tf.reshape(i, (batch_size, input_dim)) for i in 
                   tf.split(0, time_batch_len, self.seq_targets)]

        # rnn outputs a list of [batch_size x H] outputs
        outputs, self.final_state = rnn.rnn(cell, inputs, 
                                            initial_state=self.initial_state,
                                            sequence_length=self.seq_input_lengths)

        # reshape outputs (batch_size * time_batch_len) x H
        outputs_concat = tf.reshape(tf.concat(1, outputs), 
                                    [batch_size * time_batch_len, hidden_size]) 
        # reshape targets into (batch_size * time_batch_len) x D
        self.targets_concat = tf.reshape(tf.concat(1, targets),
                                         [batch_size * time_batch_len, input_dim])
        # calculate outputs of (batch_size * time_batch_len) x D
        logits = tf.matmul(outputs_concat, output_W) + output_b
        
        # probabilities of each note
        self.probs = self.calculate_probs(logits)

        self.loss = self.init_loss(logits, self.targets_concat)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay) \
                            .minimize(self.loss)

    def init_loss(self, outputs, targets):
        concat_losses = tf.nn.sigmoid_cross_entropy_with_logits(outputs, targets) 
        losses = tf.reshape(concat_losses, [self.time_batch_len, self.batch_size, self.input_dim])
        loss_per_seq = tf.reduce_sum(losses, [0, 2])
        seq_length_norm = tf.div(loss_per_seq, self.unrolled_lengths)
        return tf.reduce_sum(seq_length_norm) / self.batch_size

    def calculate_probs(self, logits):
        return tf.sigmoid(logits)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_lr_decay(self, session, lr_decay_value):
        session.run(tf.assign(self.lr_decay, lr_decay_value))


class NottinghamModel(Model):

    def init_loss(self, outputs, targets):

        #TODO: add alpha and beta
        melody_coeff = 1
        harmony_coeff = 1

        melody_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
                outputs[:, :nottingham_util.NOTTINGHAM_MELODY_RANGE],
                tf.argmax(targets[:, :nottingham_util.NOTTINGHAM_MELODY_RANGE], 1))
        harmony_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
                outputs[:, nottingham_util.NOTTINGHAM_MELODY_RANGE:],
                tf.argmax(targets[:, nottingham_util.NOTTINGHAM_MELODY_RANGE:], 1))

        concat_losses = tf.add(melody_coeff * melody_loss, harmony_coeff * harmony_loss)
        losses = tf.reshape(concat_losses, [self.time_batch_len, self.batch_size])
        loss_per_seq = tf.reduce_sum(losses, [0])
        seq_length_norm = tf.div(loss_per_seq, self.unrolled_lengths)
        return tf.reduce_sum(seq_length_norm) / self.batch_size

    def calculate_probs(self, logits):
        melody_softmax = tf.nn.softmax(logits[:, :nottingham_util.NOTTINGHAM_MELODY_RANGE])
        harmony_softmax = tf.nn.softmax(logits[:, nottingham_util.NOTTINGHAM_MELODY_RANGE:])
        return tf.concat(1, [melody_softmax, harmony_softmax]) 
