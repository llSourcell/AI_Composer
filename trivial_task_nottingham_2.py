import os, sys
import argparse
import time
 
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import nottingham_util
import sampling
import util
from model import NottinghamModel
from rnn import DefaultConfig

TICKS_PER_QUARTER = 480

if __name__ == '__main__':
    np.random.seed(1)      

    config = DefaultConfig()
    max_repeats = 5
    batch_size = 100
    minibatch_size = 100
    time_step = 120
    config.time_batch_len = time_batch_len = 100
    config.melody_coeff = melody_coeff = 0.5

    config.num_layers = 1
    config.learning_rate = lr = 1e-3
    config.learning_rate_decay = lr_decay = 0.9
    config.num_epochs = max_epochs = 500
    config.input_dropout_prob = 0.5
    config.dropout_prob = 0.5
    loss_convergence = 0.01

    chord_to_idx = {
        "CM": 0,
        "Am": 1,
        "FM": 2,
        "GM": 3
    }
    dims = nottingham_util.NOTTINGHAM_MELODY_RANGE + len(chord_to_idx)
    config.input_dim = dims

    # reshape to a (seq_length x num_dims)
    # bunch of variable length progressions
    sequences = list()
    for i in range(batch_size):

        note_length = (TICKS_PER_QUARTER/time_step)

        # chords follow a 0, 1, 2, 3 pattern
        chords = list()
        for chord in [0, 1, 2, 3]:
            chords += [chord] * note_length * 4
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
                melody_encoded += [encoding] * note_length
            encoding = np.zeros(dims)
            encoding[nottingham_util.NOTTINGHAM_MELODY_RANGE-1] = 1
            melody_encoded += [encoding] * note_length

        num_repeats = np.random.choice(np.arange(1, max_repeats))
        chord_seq = [np.add(a, b) for (a, b) in zip(chords_encoded, melody_encoded)]
        chord_seq *= num_repeats
        full_seq = np.reshape(chord_seq, [-1, dims])
        
        shift = np.random.choice(np.arange(1, full_seq.shape[0]))
        full_seq = np.roll(full_seq, shift, axis=0)

        sequences += [full_seq.copy()]

    writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=True)
    for i in range(5):
        writer.dump_sequence_to_midi(sequences[i], 
            "trivial_truth_{}.midi".format(i), time_step=time_step, resolution=TICKS_PER_QUARTER)

    data = \
        util.batch_data(sequences, time_batch_len = time_batch_len, max_time_batches = -1, softmax=True)
    #
    # assert len(notes) == len(targets) == len(rolled_lengths)
    # assert notes[0].shape[1] == len(unrolled_lengths)
    #
    # for i in range(notes[0].shape[0]):
    #     num_notes = len(np.nonzero(notes[0][i, 0, :])[0])
    #     assert num_notes == 1 or num_notes == 2

    # verification
    # total_time_steps = len(notes) * notes[0].shape[0]
    # for seq_idx, length in enumerate(unrolled_lengths):
    #     for i in range(1, total_time_steps):
    #         if i < length:
    #             assert np.array_equal(targets[(i-1)/time_batch_len][(i-1)%time_batch_len, seq_idx, :],
    #                                   notes[i/time_batch_len][i%time_batch_len, seq_idx, :])
    #         elif i == length:
    #             assert np.array_equal(targets[i/time_batch_len][i%time_batch_len, seq_idx, :],
    #                                   notes[i/time_batch_len][i%time_batch_len, seq_idx, :])
    #
    #         else:
    #             assert np.array_equal(targets[i/time_batch_len][i%time_batch_len, seq_idx, :],
    #                                   notes[i/time_batch_len][i%time_batch_len, seq_idx, :])

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("trivial", reuse=None):
            train_model = NottinghamModel(config, training=True)

        tf.initialize_all_variables().run()

        # training
        start_time = time.time()
        for i in range(max_epochs):
            loss = util.run_epoch(session, train_model, data, training=True, batch_size = minibatch_size)
            if i % 10 == 0 and i != 0:
                print 'Epoch: {}, Loss: {}, Time Per Epoch: {}'.format(i, loss, (time.time() - start_time)/i)
            if loss < loss_convergence:
                break

        # TESTING
        # with tf.variable_scope("trivial", reuse=True):
        #     test_model = NottinghamModel(config)
        #     # test_model.assign_melody_coeff(session, melody_coeff)
        #     loss, prob_vals = util.run_epoch(session, test_model, full_data,
        #         training=False, testing=True, batch_size = minibatch_size)
        #     print 'Test Loss (should be low): {}'.format(loss)
        #     util.accuracy(prob_vals, targets, unrolled_lengths, config)

        # SAMPLING SESSION #
        with tf.variable_scope("trivial", reuse=True):
            sample_model = NottinghamModel(dict(config, **{
                "time_batch_len": 1
            }), training=False)

        # start with the first chord
        # chord = sequences[0][0]
        chord = np.zeros(dims)
        seq = [chord]
        state = sample_model.get_cell_zero_state(session, 1)
        sampler = nottingham_util.NottinghamSampler(chord_to_idx, verbose=False)

        for i in range(note_length * 16 * max_repeats * 2):
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
        writer.dump_sequence_to_midi(seq, "trivial.midi", time_step=time_step, resolution=TICKS_PER_QUARTER)
        print 'Final Train Loss: {}'.format(loss)
