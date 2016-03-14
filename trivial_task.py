import os, sys
import argparse
import time
 
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import sampling
import util
from model import Model

if __name__ == '__main__':
    np.random.seed(1)      
    
    dims = midi_util.RANGE
    max_repeats = 10
    batch_size = 100
    minibatch_size = 78
    time_batch_len = 8

    lr = 1e-2
    lr_decay = 0.9
    max_epochs = 1000
    loss_convergence = 0.0001

    # reshape to a (seq_length x num_dims)
    # bunch of variable length progressions
    sequences = list()
    for i in range(batch_size):
        num_repeats = np.random.choice(np.arange(1, max_repeats))
        chord_seq = midi_util.i_vi_iv_v(num_repeats)
        chord_seq = np.reshape(chord_seq, [-1, dims])
        sequences += [chord_seq.copy()]

    notes, targets, rolled_lengths, unrolled_lengths = \
        util.batch_data(sequences, time_batch_len = time_batch_len, max_time_batches = -1)

    assert len(notes) == len(targets) == len(rolled_lengths)
    assert notes[0].shape[1] == len(unrolled_lengths)

    full_data = {
        "data": notes,
        "targets":  targets,
        "seq_lengths": rolled_lengths,
        "unrolled_lengths": unrolled_lengths
    }

    config = {
        "input_dim": dims,
        "hidden_size": 100,
        "num_layers": 1,
        "dropout_prob": 1.0,
        "time_batch_len": time_batch_len,
        "cell_type": "lstm"
    } 

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("trivial", reuse=None):
            train_model = Model(config, training=True)

        tf.initialize_all_variables().run()

        # # training
        train_model.assign_lr(session, lr)
        train_model.assign_lr_decay(session, lr_decay)
        time_start = time.time()
        for i in range(max_epochs):
            loss = util.run_epoch(session, train_model, full_data, training=True,
                                  batch_size = minibatch_size)
            if i % 10 == 0 and i != 0:
                print 'Epoch {}, Loss: {}, Time Per Epoch {}'.\
                    format(i, loss, (time.time() - time_start) / i)
            if loss < loss_convergence:
                break

        # TESTING
        with tf.variable_scope("trivial", reuse=True):
            test_model = Model(config)
            loss, prob_vals = util.run_epoch(session, test_model, full_data,
                training=False, testing=True, batch_size = minibatch_size)
            print 'Test Loss (should be low): {}'.format(loss)
            util.accuracy(prob_vals, targets, unrolled_lengths, config)
        
        # SAMPLING SESSION #
        with tf.variable_scope("trivial", reuse=True):
            sample_model = Model(dict(config, **{
                "time_batch_len": 1
            }), training=False)

        # start with the first chord
        chord = midi_util.cmaj()
        seq = [chord]
        state = sample_model.get_cell_zero_state(session, 1)
        sampler = sampling.Sampler(verbose=False)

        for i in range(4 * max_repeats * 2):
            chord = np.reshape(chord, [1, 1, dims])
            feed = {
                sample_model.seq_input: chord,
                sample_model.initial_state: state,
                sample_model.seq_input_lengths: [1],
            }
            [probs, state] = session.run(
                [sample_model.probs, sample_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, dims)
            chord = sampler.sample_notes(probs)
            seq.append(chord)

        writer = midi_util.MidiWriter(verbose=True) 
        writer.dump_sequence_to_midi(seq, "trivial.midi", time_step=120, resolution=100)
