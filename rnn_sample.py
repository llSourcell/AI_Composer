import os, sys
import argparse
import time
import itertools
import cPickle

import numpy as np
import tensorflow as tf    

import util
import nottingham_util
from model import Model, NottinghamModel
from rnn import DefaultConfig

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Script to generated a MIDI file sample from a trained model.')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--sample_seq', type=str, default='random',
        choices = ['random', 'chords'])
    parser.add_argument('--conditioning', type=int, default=-1)
    parser.add_argument('--sample_length', type=int, default=512)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f: 
        config = cPickle.load(f)

    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        model_class = NottinghamModel
        with open(nottingham_util.PICKLE_LOC, 'r') as f:
            pickle = cPickle.load(f)
        chord_to_idx = pickle['chord_to_idx']

        time_step = 120
        resolution = 480

        # use time batch len of 1 so that every target is covered
        test_data = util.batch_data(pickle['test'], time_batch_len = 1, 
            max_time_batches = -1, softmax = True)
    else:
        raise Exception("Other datasets not yet implemented")

    print config

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)

        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        saver.restore(session, model_path)

        state = sampling_model.get_cell_zero_state(session, 1)
        if args.sample_seq == 'chords':
            # 16 - one measure, 64 - chord progression
            repeats = args.sample_length / 64
            sample_seq = nottingham_util.i_vi_iv_v(chord_to_idx, repeats, config.input_dim)
            print 'Sampling melody using a I, VI, IV, V progression'

        elif args.sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(pickle['test'])))
            sample_seq = [ pickle['test'][sample_index][i, :] 
                for i in range(pickle['test'][sample_index].shape[0]) ]

        chord = sample_seq[0]
        seq = [chord]

        if args.conditioning > 0:
            for i in range(1, args.conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord = sample_seq[i]
                seq.append(chord)

        if config.dataset == 'softmax':
            writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
            sampler = nottingham_util.NottinghamSampler(chord_to_idx, verbose=False)
        else:
            # writer = midi_util.MidiWriter()
            # sampler = sampling.Sampler(verbose=False)
            raise Exception("Other datasets not yet implemented")

        for i in range(max(args.sample_length - len(seq), 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [config.input_dim])
            chord = sampler.sample_notes(probs)

            if config.dataset == 'softmax':
                r = nottingham_util.NOTTINGHAM_MELODY_RANGE
                if args.sample_melody:
                    chord[r:] = 0
                    chord[r:] = sample_seq[i][r:]
                elif args.sample_harmony:
                    chord[:r] = 0
                    chord[:r] = sample_seq[i][:r]

            seq.append(chord)

        writer.dump_sequence_to_midi(seq, "best.midi", 
            time_step=time_step, resolution=resolution)
