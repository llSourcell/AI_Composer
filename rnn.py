import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    
from pprint import pprint
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    

from midi_util import midi_to_sequence, sequence_to_midi, i_vi_iv_v
from preprocess import load_data
from model import Model

def run_epoch(session, model, seq_data, seq_targets, verbose=False):
    _, loss_value = session.run(
        [model.train_step, model.loss],
        feed_dict={
            model.seq_input: seq_data,
            model.seq_targets: seq_targets
        })
    return loss_value

if __name__ == '__main__':
    np.random.seed(1)      


    # N = 1000
    # D = 127
    # L = 3
    # 0, 1, 2, 3 and end with 4 "end of sequence"
    # train_seqs = [[[0], [1], [2], [3]] * L] * 200
    # train_len = 4 * L
    # train_seqs = [i_vi_iv_v(3)] * 200
    # train_len = len(train_seqs[0])
    # train_data, train_targets = sequences_to_data(train_seqs, train_len, D)

    data = load_data()
    print 'Finished loading data'

    default_config = {
        "input_dim": data["input_dim"],
        "hidden_size": 100,
        "num_layers": 2,
    } 

    def set_config(name):
        return dict(default_config, **{
            "batch_size": data[name]["data"].shape[1],
            "seq_length": data[name]["data"].shape[0]
        })

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope("model", reuse=None):
            train_model = Model(set_config("train"))
        with tf.variable_scope("model", reuse=True):
            valid_model = Model(set_config("valid"))
            test_model = Model(set_config("test"))

        saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()

        # training
        num_epochs = 100
        learning_rate = 0.01
        train_model.assign_lr(session, learning_rate)
        for i in range(100):
            loss = run_epoch(session, train_model, train_data, train_targets)
            if i % 5 == 0:
                print 'Epoch: {}, Loss: {}'.format(i, loss)
        
        # testing
        loss = run_epoch(session, test_model, test_data, test_targets)
        print 'Testing loss: {}'.format(loss)

        # sampling
        def sample_notes_from_probs(probs, n):
            reshaped = np.reshape(probs, [D])
            top_idxs = reshaped.argsort()[-n:][::-1]
            chord = np.zeros([D], dtype=np.int32)
            chord[top_idxs] = 1.0
            return chord

        def visualize_probs(probs, targets):
            probs = list(probs[0])
            targets = list(targets)

            # print 'First four notes: '
            # pprint(zip(probs, targets))
            # print 'Highest four probs: '
            # pprint(sorted(list(enumerate(probs)), key=lambda x: x[1], 
            #                                       reverse=True)[:4])

        # start with the first chord
        with tf.variable_scope("model", reuse=True):
            sampling_model = Model(dict(default_config, **{
                "batch_size": 1,
                "seq_length": 1
            }))

        chord = train_data[0, 0, :]
        seq = [chord]
        state = sampling_model.cell.zero_state(sampling_model.N, tf.float32) \
                              .eval(session=session)

        for i in range(train_len):
            seq_input = np.reshape(chord, [1, 1, D])
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            visualize_probs(probs, train_targets[i, 0, :])
            print probs
            chord = sample_notes_from_probs(probs, 4)
            seq.append(chord)
         
        chords = [np.nonzero(c)[0].tolist() for c in seq]
        sequence_to_midi(chords, "gen1.midi")


        saver.save(session, os.path.join("models", "jsb_chorales.model"))

