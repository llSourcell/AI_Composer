import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq

import nottingham_util

class Model(object):
    """ RNN Model """
    
    def __init__(self, config, training=False):

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

        self.cell = rnn_cell.MultiRNNCell([first_layer] + [hidden_layer] * (1-num_layers))
        if training and dropout_prob < 1.0:
            self.cell = rnn_cell.DropoutWrapper(self.cell, output_keep_prob = dropout_prob)

        self.seq_input = \
            tf.placeholder(tf.float32, shape=[time_batch_len, None, input_dim])
        self.seq_input_lengths = \
            tf.placeholder(tf.int32, [None])
        self.unrolled_lengths = \
            tf.placeholder(tf.float32, [None])

        batch_size = tf.shape(self.seq_input)[0]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        inputs_list = tf.unpack(self.seq_input)

        # rnn outputs a list of [batch_size x H] outputs
        outputs_list, self.final_state = rnn.rnn(self.cell, inputs_list, 
                                                 initial_state=self.initial_state,
                                                 sequence_length=self.seq_input_lengths)

        # logits = tf.pack([tf.matmul(outputs_list[t], output_W) + output_b for t in range(time_batch_len)])

        # TODO: verify if the below is faster and correct
        outputs = tf.pack(outputs_list)
        # outputs_concat = tf.reshape(outputs, [time_batch_len * batch_size, hidden_size])
        outputs_concat = tf.reshape(outputs, [-1, hidden_size])
        logits_concat = tf.matmul(outputs_concat, output_W) + output_b
        logits = tf.reshape(logits_concat, [time_batch_len, -1, input_dim])

        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logits_concat)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay) \
                            .minimize(self.loss)

    def init_loss(self, outputs, _):
        self.seq_targets = \
            tf.placeholder(tf.float32, [self.time_batch_len, None, self.input_dim])

        batch_size = tf.shape(self.seq_input)

        # losses = tf.nn.sigmoid_cross_entropy_with_logits(outputs, self.seq_targets)
        # self.concat_losses = tf.reduce_sum(losses, [0, 2])
        # seq_length_norm = tf.div(self.concat_losses, self.unrolled_lengths)
        # return tf.reduce_sum(seq_length_norm) / tf.to_float(batch_size)

        self.concat_losses = tf.zeros_like(self.seq_input_lengths, dtype=tf.float32)
        ts = tf.constant(0)
        for output_ts, target_ts in zip(tf.unpack(outputs), tf.unpack(self.seq_targets)):
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(output_ts, target_ts)
            by_batches = tf.reduce_sum(cross_ent, [1])
            good_seqs = tf.less(ts, self.seq_input_lengths)
            masked = tf.select(tf.less(ts, self.seq_input_lengths), by_batches, 
                tf.zeros_like(self.seq_input_lengths, dtype=tf.float32))
            masked = tf.div(masked, tf.to_float(tf.maximum(self.seq_input_lengths, 
                                                tf.ones_like(self.seq_input_lengths))))

            self.concat_losses = tf.add(self.concat_losses, masked)
            ts = tf.add(ts, 1)

        return tf.reduce_sum(self.concat_losses) / tf.to_float(batch_size[0])

    def calculate_probs(self, logits):
        return tf.sigmoid(logits)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_lr_decay(self, session, lr_decay_value):
        session.run(tf.assign(self.lr_decay, lr_decay_value))

    def get_cell_zero_state(self, session, batch_size):
        return self.cell.zero_state(batch_size, tf.float32).eval(session=session)

class NottinghamModel(Model):

    def init_loss(self, outputs, outputs_concat):
        self.seq_targets = \
            tf.placeholder(tf.int64, [self.time_batch_len, None, 2])
        batch_size = tf.shape(self.seq_targets)[1]

        with tf.variable_scope("rnnlstm"):
            self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)
            self.lr_decay = tf.Variable(0.0, name="learning_rate_decay", trainable=False)
            self.melody_coeff = tf.Variable(0.5, name="melody_coeff", trainable=False)

        r = nottingham_util.NOTTINGHAM_MELODY_RANGE
        targets_concat = tf.reshape(self.seq_targets, [-1, 2])
        melody_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            outputs_concat[:, :r], \
            targets_concat[:, 0])
        harmony_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
            outputs_concat[:, r:], \
            targets_concat[:, 1])
        losses = tf.add(self.melody_coeff * melody_loss, (1 - self.melody_coeff) * harmony_loss)
        self.concat_losses = tf.reduce_sum(tf.reshape(losses, [self.time_batch_len, -1]), 0)

        # TODO: is the method below slower?
        # melody_loss, harmony_loss = None, None
        # for t in range(self.time_batch_len):
        #     mloss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
        #         outputs[t, :, :nottingham_util.NOTTINGHAM_MELODY_RANGE], \
        #         tf.argmax(targets[t, :, :nottingham_util.NOTTINGHAM_MELODY_RANGE], 1))
        #     if t == 0:
        #         melody_loss = mloss
        #     else:
        #         melody_loss = tf.add(mloss, melody_loss)
        #
        #     hloss = tf.nn.sparse_softmax_cross_entropy_with_logits( \
        #         outputs[t, :, nottingham_util.NOTTINGHAM_MELODY_RANGE:], \
        #         tf.argmax(targets[t, :, nottingham_util.NOTTINGHAM_MELODY_RANGE:], 1))
        #     if t == 0:
        #         harmony_loss = hloss
        #     else:
        #         harmony_loss = tf.add(hloss, harmony_loss)
        # concat_losses = tf.add(self.melody_coeff*melody_loss, (1-self.melody_coeff)*harmony_loss)

        seq_length_norm = tf.div(self.concat_losses, 
            tf.to_float(tf.maximum(self.seq_input_lengths, tf.ones_like(self.seq_input_lengths))))
        return tf.reduce_sum(seq_length_norm) / tf.to_float(batch_size)

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
