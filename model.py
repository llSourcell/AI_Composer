import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq

import ipdb

class Model(object):
    """ RNN Model """
    
    def __init__(self, config, training=False):

        self.batch_size = batch_size = config["batch_size"]
        self.time_batch_len = time_batch_len = config["time_batch_len"]
        input_dim = config["input_dim"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        dropout_prob = config["dropout_prob"]
        cell_type = config["cell_type"]

        # print config

        if (dropout_prob <= 0.0 or dropout_prob > 1.0):
            raise Exception("Invalid dropout probability: {}".format(dropout_prob))

        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [hidden_size, input_dim])
            output_b = tf.get_variable("output_b", [input_dim])
            self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)
            self.lr_decay = tf.Variable(0.0, name="learning_rate_decay", trainable=False)

        if cell_type == "vanilla":
            base_cell = rnn_cell.BasicRNNCell(hidden_size, input_size = input_dim)  
        else:
            base_cell = rnn_cell.BasicLSTMCell(hidden_size, input_size = input_dim)  

        cell = rnn_cell.MultiRNNCell([base_cell] * num_layers)
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

        # TODO: change to use dynamic_rnn?
        # rnn outputs max_seq_len x batch_size x hidden_size
        # outputs, self.final_state = \
        #     rnn.dynamic_rnn(cell, self.seq_input, 
        #                     initial_state=self.initial_state,
        #                     sequence_length=self.seq_input_lengths,
        #                     time_major=False,
        #                     dtype=tf.float32)
        # outputs_concat = tf.reshape(outputs,
        #                             [batch_size * time_batch_len, hidden_size]) 

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
        self.targets_concat = tf.reshape(self.seq_targets,
                                         [batch_size * time_batch_len, input_dim])
        # calculate outputs of (batch_size * time_batch_len) x D
        outputs = tf.matmul(outputs_concat, output_W) + output_b
        
        # probabilities of each note
        self.probs = tf.sigmoid(outputs)

        concat_losses = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.targets_concat)
        loss_per_time_step = tf.reduce_sum(concat_losses, 1)
        loss_per_seq = tf.reduce_sum(tf.reshape(loss_per_time_step, [batch_size, time_batch_len]), 1)
        seq_length_norm = tf.div(loss_per_seq, self.unrolled_lengths)
        self.loss = tf.reduce_sum(seq_length_norm) / batch_size

        # ipdb.set_trace()
        # self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay) \
                            .minimize(self.loss)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_lr_decay(self, session, lr_decay_value):
        session.run(tf.assign(self.lr_decay, lr_decay_value))
