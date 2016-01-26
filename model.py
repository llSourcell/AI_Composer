import os
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn, seq2seq
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    

LEARNING_RATE = 0.1
INIT_RANGE = 0.1
EPOCHS = 100
H = 100

class Model(object):
    
    def __init__(self, batch_size, input_dimension, seq_length, 
                 hidden_size=100, initialization=0.1,
                 sampling_mode=False):

        if sampling_mode:
            self.N = 1
            self.seq_length = 1
        else:
            self.N = batch_size
            self.seq_length = seq_length

        self.D = input_dimension
        self.H = hidden_size
        self.init_range = initialization
        
        self.initializer = tf.random_uniform_initializer(-self.init_range, 
                                                         self.init_range) 
        self.seq_input = tf.placeholder(tf.float32, [self.seq_length, self.N, self.D])
        self.seq_targets = tf.placeholder(tf.float32, [self.seq_length, self.N, self.D])

        self.cell = LSTMCell(self.H, self.D, initializer=self.initializer)  
        self.initial_state = self.cell.zero_state(self.N, tf.float32)

        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [self.H, self.D], initializer=self.initializer)
            output_b = tf.get_variable("output_b", [self.D], initializer=self.initializer)
            self.learning_rate = tf.Variable(0.0, name="learning_rate", trainable=False)

        # make inputs a list of [N x D] time-steps
        inputs = [tf.reshape(i, (self.N, self.D)) for i in 
                  tf.split(0, self.seq_length, self.seq_input)]
        # rnn outputs a list of [N x H] outputs
        hiddens, states = rnn.rnn(self.cell, inputs, initial_state=self.initial_state)
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
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, seq_input, seq_targets, num_epochs=100, learning_rate=0.1):

        saver = tf.train.Saver(tf.all_variables())

        # initialize all variables 
        self.session = tf.Session()
        iop = tf.initialize_all_variables()
        self.session.run(iop)

        # epochs
        for i in range(num_epochs):
            feed = {self.seq_input: seq_input,
                    self.seq_targets: seq_targets,
                    self.learning_rate: learning_rate}
            _, loss_value = self.session.run([self.train_step, self.loss], feed_dict=feed)
            print 'Epoch: {}, Loss: {}'.format(i, loss_value)

        print 'Done training. Saving checkpoint file.'
        model_path = os.path.join("models", "jsb_chorale.model")
        saver.save(self.session, model_path)

    def load_model(self, path):
        self.session = tf.Session()
        iop = tf.initialize_all_variables()
        self.session.run(iop)
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(self.session, path)

    def test(self, seq_input, seq_targets):
        self.load_model(os.path.join("models", "jsb_chorale.model"))

        feed = {self.seq_input: seq_input,
                self.seq_targets: seq_targets}
        loss = self.session.run(self.loss, feed_dict=feed)
        print 'Testing loss: {}'.format(loss)
        return loss


    def sample_notes_from_probs(self, probs):
        reshaped = np.reshape(probs, [self.D])
        top_idxs = np.argpartition(reshaped, 4)[-4:]
        chord = np.zeros([self.D], dtype=np.float32)
        chord[top_idxs] = 1.0
        return chord

    def sample(self, starting_chord, seq_length=100, temperature=0.5):

        self.load_model(os.path.join("models", "jsb_chorale.model"))

        seq = [starting_chord]
        chord = starting_chord
        state = self.cell.zero_state(self.N, tf.float32).eval(session=self.session)

        for i in range(seq_length):
            seq_input = np.reshape(chord, [1, 1, self.D])
            feed = {self.seq_input: seq_input,
                    self.initial_state: state}
            [probs, state] = self.session.run([self.probs, self.final_state], 
                                              feed_dict=feed)
            chord = self.sample_notes_from_probs(probs)
            seq.append(chord)

        return seq
