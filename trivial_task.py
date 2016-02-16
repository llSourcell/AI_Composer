import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import sampling
import util
from model import Model

if __name__ == '__main__':
    np.random.seed(1)      
    
    dims = midi_util.RANGE*2
    num_repeats = 5
    batch_size = 250

    lr = 1e-2
    lr_decay = 0.9
    max_epochs = 1000
    loss_convergence = 0.5

    # midi_util.dump_sequence_to_midi(chord_seq, "trivial.midi")

    chord_seq = midi_util.i_vi_iv_v(num_repeats)
    # reshape to a (seq_length x num_dims)
    chord_seq = np.reshape(chord_seq, [-1, dims])
    # duplicate to (batch_size x seq_length x num_dims)
    chord_seq = np.tile(chord_seq, (batch_size, 1, 1))
    # swap axis for (seq_length x batch_size x num_dims)
    data = np.swapaxes(chord_seq, 0, 1)
    targets = util.prepare_targets(data)

    config = {
        "input_dim": dims,
        "hidden_size": 100,
        "num_layers": 1,
        "dropout_prob": 1.0,
        "batch_size": data.shape[1],
        "seq_length": data.shape[0]
    } 

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("trivial", reuse=None):
            train_model = Model(config, training=True)

        tf.initialize_all_variables().run()

        # training
        train_model.assign_lr(session, lr)
        train_model.assign_lr_decay(session, lr_decay)
        for i in range(max_epochs):
            loss = util.run_epoch(session, train_model, data, targets, training=True)
            if i % 10 == 0:
                print 'Loss: {}'.format(loss)
            if loss < loss_convergence:
                break

        # SAMPLING SESSION #
        with tf.variable_scope("trivial", reuse=True):
            sample_model = Model(dict(config, **{
                "batch_size": 1,
                "seq_length": 1
            }), training=False)
          
        # start with the first chord
        chord = midi_util.cmaj()
        seq = [chord]
        state = sample_model.initial_state.eval()
        sampler = sampling.Sampler()

        for i in range(data.shape[0]+1):
            chord = np.reshape(chord, [1, 1, dims])
            feed = {
                sample_model.seq_input: chord,
                sample_model.initial_state: state
            }
            [probs, state] = session.run(
                [sample_model.probs, sample_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, dims)
            chord = sampler.sample_notes_prob(probs)
            seq.append(chord)
    
        midi_util.dump_sequence_to_midi(seq, "trivial.midi")
