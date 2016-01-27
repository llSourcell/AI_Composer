import os
 
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    

from midi_util import midi_to_sequence, sequence_to_midi
from model import Model

DATA_DIR = 'data/JSBChorales'

HIDDEN_SIZE = 100

def prepare_data(dataset_name):
    train_path = os.path.join(DATA_DIR, dataset_name)
    train_files = [ os.path.join(train_path, f) for f in os.listdir(train_path)
                  if os.path.isfile(os.path.join(train_path, f)) ] 
    train_seq = [ midi_to_sequence(f) for f in train_files ]

    # number of examples 
    N = len(train_seq)
    # sequence length is the longest sequence
    seq_length = max(len(t) for t in train_seq)
    # dimension of input
    max_note = max(max([max(c) for c in seq if len(c) > 0]) for seq in train_seq)
    min_note = min(min([min(c) for c in seq if len(c) > 0]) for seq in train_seq)

    return (train_seq, seq_length, max_note)

def sequences_to_data(seqs, max_seq_length, D):
    # (sequence length) X (num sequences) X (dimension)
    data = np.zeros([max_seq_length, len(seqs), D], dtype=np.float32)
    for idx, seq in enumerate(seqs):
      for c_idx, chord in enumerate(seq):
          data[c_idx, idx, chord] = 1

    # roll back the time steps axis to get the target of each example
    targets = np.roll(data, -1, axis=0)
    # set final time step targets to 0
    targets[-1, :, :] = 0 

    return (data, targets)

def run_epoch(session, model, seq_data, seq_targets):
    _, loss_value = session.run(
        [model.train_step, model.loss],
        feed_dict={
            model.seq_input: seq_data,
            model.seq_targets: seq_targets
        })
    return loss_value

if __name__ == '__main__':
    np.random.seed(1)      

    train_seqs, train_len, train_max = prepare_data('train')
    valid_seqs, valid_len, valid_max = prepare_data('valid')
    test_seqs, test_len, test_max = prepare_data('valid')

    D = max(train_max, valid_max, test_max)+1

    print 'Dimension: {}'.format(D)

    train_data, train_targets = sequences_to_data(train_seqs, train_len, D)
    test_data, test_targets = sequences_to_data(test_seqs, test_len, D)
    valid_data, valid_targets = sequences_to_data(valid_seqs, valid_len, D)

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope("model", reuse=None):
            train_model = Model(len(train_seqs), D, train_len, hidden_size=HIDDEN_SIZE)
        with tf.variable_scope("model", reuse=True):
            valid_model = Model(len(valid_seqs), D, valid_len, hidden_size=HIDDEN_SIZE)
            test_model = Model(len(test_seqs), D, test_len, hidden_size=HIDDEN_SIZE)

        tf.initialize_all_variables().run()

        # training
        num_epochs = 25
        learning_rate = 0.1
        train_model.assign_lr(session, learning_rate)
        for i in range(num_epochs):
            loss = run_epoch(session, train_model, train_data, train_targets)
            if i % 5 == 0:
                print 'Epoch: {}, Loss: {}'.format(i, loss)
        
        # testing
        loss = run_epoch(session, test_model, test_data, test_targets)
        print 'Testing loss: {}'.format(loss)

        # sampling
        def sample_notes_from_probs(probs):
            reshaped = np.reshape(probs, [D])
            top_idxs = np.argpartition(reshaped, 4)[-4:]
            chord = np.zeros([D], dtype=np.float32)
            chord[top_idxs] = 1.0
            return chord

        seq_length = 100
        # start with the first chord
        with tf.variable_scope("model", reuse=True):
            sampling_model = Model(1, D, 1, hidden_size=HIDDEN_SIZE)

        chord = train_data[0, 0, :]
        seq = [chord]
        state = sampling_model.cell.zero_state(sampling_model.N, tf.float32) \
                              .eval(session=session)

        for i in range(seq_length):
            seq_input = np.reshape(chord, [1, 1, D])
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            chord = sample_notes_from_probs(probs)
            print chord
            print state
            seq.append(chord)
         
        chords = [np.nonzero(c)[0].tolist() for c in seq]
        sequence_to_midi(chords, "gen1.midi")
