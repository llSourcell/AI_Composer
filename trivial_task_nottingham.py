import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import nottingham_util
import sampling
import util
from model import NottinghamModel

if __name__ == '__main__':
    np.random.seed(1)      
    
    max_repeats = 5
    batch_size = 100

    lr = 1e-3
    lr_decay = 0.9
    max_epochs = 1000
    loss_convergence = 0.1

    chord_to_idx = {
        "CM": 0,
        "Am": 1,
        "FM": 2,
        "GM": 3
    }
    dims = nottingham_util.NOTTINGHAM_MELODY_RANGE + len(chord_to_idx)

    # reshape to a (seq_length x num_dims)
    # bunch of variable length progressions
    sequences = list()
    for i in range(batch_size):

        # chords follow a 0, 1, 2, 3 pattern
        chords = list()
        for chord in [0, 1, 2, 3]:
            chords += [chord] * 4
        chords_encoded = list()
        for chord in chords:
            encoding = np.zeros(dims)
            encoding[chord + nottingham_util.NOTTINGHAM_MELODY_RANGE] = 1
            chords_encoded.append(encoding)

        # quarter notes are scales
        melody_encoded = list()
        for chord in [[72, 76, 79], [72, 76, 81], [72, 77, 81], [74, 79, 83]]: 
            for note in chord:
                encoding = np.zeros(dims)
                encoding[note - nottingham_util.NOTTINGHAM_MELODY_MIN] = 1
                melody_encoded.append(encoding)
            melody_encoded.append(np.zeros(dims))


        num_repeats = np.random.choice(np.arange(1, max_repeats))
        chord_seq = [np.add(a, b) for (a, b) in zip(chords_encoded, melody_encoded)]
        chord_seq *= num_repeats
        full_seq = np.reshape(chord_seq, [-1, dims])
        sequences += [full_seq.copy()]

    writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=True)
    writer.dump_sequence_to_midi(sequences[0], "trivial_truth.midi", time_step=480, resolution=480)

    notes, targets, rolled_lengths, unrolled_lengths = util.batch_data(sequences, time_batch_len = 4, max_time_batches = -1)
    # notes, targets, rolled_lengths, unrolled_lengths = util.batch_data(sequences, time_batch_len = -1, max_time_batches = -1)

    assert len(notes) == len(targets) == len(rolled_lengths)
    assert notes[0].shape[1] == len(unrolled_lengths)
    
    for i in range(notes[0].shape[0]):
        num_notes = len(np.nonzero(notes[0][i, 0, :])[0])
        assert num_notes == 1 or num_notes == 2

    assert np.array_equal(notes[0][1, 0, :], targets[0][0, 0, :])
    assert np.array_equal(notes[0][2, 0, :], targets[0][1, 0, :])
    assert np.array_equal(notes[0][3, 0, :], targets[0][2, 0, :])
    assert np.array_equal(notes[1][0, 0, :], targets[0][3, 0, :])

    # writer.dump_sequence_to_midi(notes[0][:, 0, :], "trivial_truth.midi", time_step=480, resolution=480)
    # print notes
    # print targets
    # print rolled_lengths
    # print unrolled_lengths
    #
    # sys.exit(0)

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
        "batch_size": batch_size,
        "time_batch_len": 4,
        "cell_type": "lstm"
    } 

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("trivial", reuse=None):
            train_model = NottinghamModel(config, training=True)

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
            sample_model = NottinghamModel(dict(config, **{
                "batch_size": 1,
                "time_batch_len": 1
            }), training=False)

        # start with the first chord
        chord = sequences[0][0]
        seq = [chord]
        state = sample_model.initial_state.eval()
        sampler = nottingham_util.NottinghamSampler(verbose=True)

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
            chord = sampler.sample_notes(probs)
            seq.append(chord)

        writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=True)
        writer.dump_sequence_to_midi(seq, "trivial.midi", time_step=480, resolution=480)
