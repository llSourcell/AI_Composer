import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn, seq2seq
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    

class Model(object):
    
    def __init__(self, batch_size, input_dimension, seq_length, hidden_size=100):

        self.N = batch_size
        self.seq_length = seq_length
        self.D = input_dimension
        self.H = hidden_size

        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [self.H, self.D])
            output_b = tf.get_variable("output_b", [self.D])
            self.lr = tf.Variable(0.0, name="learning_rate", trainable=False)

        self.seq_input = \
            tf.placeholder(tf.float32, [self.seq_length, self.N, self.D])
        self.seq_targets = \
            tf.placeholder(tf.float32, [self.seq_length, self.N, self.D])
        self.cell = LSTMCell(self.H, self.D)  
        self.initial_state = self.cell.zero_state(self.N, tf.float32)

        # make inputs a list of [N x D] time-steps
        inputs = [tf.reshape(i, (self.N, self.D)) for i in 
                  tf.split(0, self.seq_length, self.seq_input)]
        # rnn outputs a list of [N x H] outputs
        hiddens, states = rnn.rnn(self.cell, inputs, 
                                  initial_state=self.initial_state)
        assert len(hiddens) == self.seq_length
        self.final_state = states[-1]

        # reshape into [(N * seq_length) x H]
        hiddens_concat = tf.reshape(tf.concat(1, hiddens), [-1, self.H]) 
        # calculate outputs of [(N * seq_length) x D]
        outputs = tf.matmul(hiddens_concat, output_W) + output_b

        # probabilities of each note
        self.probs = tf.sigmoid(outputs)

        # reshape targets into [(N * seq_length) x D]
        targets_concat = tf.reshape(self.seq_targets, [-1, self.D])
        # sanity check
        # assert outputs.get_shape() == targets_concat.get_shape()

        losses = tf.nn.sigmoid_cross_entropy_with_logits(outputs, targets_concat)
        self.loss = tf.reduce_sum(losses) / self.N / self.seq_length
        self.train_step = tf.train.AdagradOptimizer(self.lr) \
                                  .minimize(self.loss)
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
