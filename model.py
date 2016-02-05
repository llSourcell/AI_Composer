import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn, seq2seq

# TODO(yoavz): ideas
# 1. add dropout?

class Model(object):
    """ RNN Model """
    
    def __init__(self, config):

        self.batch_size = batch_size = config["batch_size"]
        self.seq_length = seq_length = config["seq_length"]
        input_dim = config["input_dim"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]

        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [hidden_size, input_dim])
            output_b = tf.get_variable("output_b", [input_dim])
            self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)

        self.seq_input = \
            tf.placeholder(tf.float32, [seq_length, batch_size, input_dim])
        self.seq_targets = \
            tf.placeholder(tf.float32, [seq_length, batch_size, input_dim])

        # cell = MultiBasicRNNCell(hidden_size)  
        lstm_cell = rnn_cell.BasicLSTMCell(hidden_size)  
        cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # make inputs/targets a list of [batch_size x D] time-steps
        inputs = [tf.reshape(i, (batch_size, input_dim)) for i in 
                  tf.split(0, seq_length, self.seq_input)]
        targets = [tf.reshape(i, (batch_size, input_dim)) for i in 
                   tf.split(0, seq_length, self.seq_targets)]
        # rnn outputs a list of [batch_size x H] outputs
        outputs, states = rnn.rnn(cell, inputs, 
                                  initial_state=self.initial_state)
        self.final_state = states[-1]

        # reshape into [(batch_size * seq_length) x H]
        outputs_concat = tf.reshape(tf.concat(1, outputs), 
                                    [batch_size * seq_length, hidden_size]) 
        # calculate outputs of [(batch_size * seq_length) x D]
        outputs = tf.matmul(outputs_concat, output_W) + output_b
        # reshape targets into [(batch_size * seq_length) x D]
        targets_concat = tf.reshape(tf.concat(1, targets), 
                                    [batch_size * seq_length, input_dim])

        # probabilities of each note
        self.probs = tf.sigmoid(outputs)

        losses = tf.nn.sigmoid_cross_entropy_with_logits(outputs, targets_concat)
        self.loss = tf.reduce_sum(losses) / batch_size / seq_length
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay=0.9) \
                                  .minimize(self.loss)

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
