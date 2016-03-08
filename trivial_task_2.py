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
    
    dims = midi_util.RANGE
    max_repeats = 5
    batch_size = 100

    lr = 1e-2
    lr_decay = 0.9
    max_epochs = 200
    loss_convergence = 0.3

    # reshape to a (seq_length x num_dims)
    # bunch of variable length progressions
    sequences = list()
    for i in range(batch_size):
        num_repeats = np.random.choice(np.arange(1, max_repeats))
        chord_seq = midi_util.i_vi_iv_v(1)
        chord_seq = (chord_seq + chord_seq[::-1]) * num_repeats
        full_seq = np.reshape(chord_seq, [-1, dims])
        sequences += [full_seq.copy()]

    writer = midi_util.MidiWriter(verbose=True) 
    writer.dump_sequence_to_midi(sequences[0], "trivial_truth.midi", time_step=120, resolution=100)

    notes, targets, rolled_lengths, unrolled_lengths = util.batch_data(sequences, time_batch_len = 4, max_time_batches = -1)

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
        "hidden_size": 200,
        "num_layers": 2,
        "dropout_prob": 1.0,
        "batch_size": batch_size,
        "time_batch_len": 4,
        "cell_type": "lstm"
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
            loss = util.run_epoch(session, train_model, full_data, training=True)
            if i % 10 == 0:
                print 'Loss: {}'.format(loss)
            if loss < loss_convergence:
                break

        # SAMPLING SESSION #
        with tf.variable_scope("trivial", reuse=True):
            sample_model = Model(dict(config, **{
                "batch_size": 1,
                "time_batch_len": 1
            }), training=False)

        # start with the first chord
        chord = midi_util.cmaj()
        seq = [chord]
        state = sample_model.initial_state.eval()
        sampler = sampling.Sampler(verbose=True)

        for i in range(8 * max_repeats * 2):
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
            chord = sampler.sample_notes(probs, num_notes=3)
            seq.append(chord)

        writer.dump_sequence_to_midi(seq, "trivial.midi", time_step=120, resolution=100)
