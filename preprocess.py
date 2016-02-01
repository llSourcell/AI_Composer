import os
import numpy as np
import tensorflow as tf    

from midi_util import midi_to_sequence

def prepare_data(dataset_subdir, data_dir='data/JSBChorales'):
    """ Args: location of the dataset
        Returns: 
            train_seq - list of sequences, where each sequence is a list of chords
            max_seq_length - length of longest seq
            max_note - max note value
    """
    train_path = os.path.join(data_dir, dataset_subdir)
    train_files = [ os.path.join(train_path, f) for f in os.listdir(train_path)
                  if os.path.isfile(os.path.join(train_path, f)) ] 
    train_seq = [ midi_to_sequence(f) for f in train_files ]

    # number of examples 
    N = len(train_seq)
    # sequence length is the longest sequence
    max_seq_length = max(len(t) for t in train_seq)
    # dimension of input
    max_note = max(max([max(c) for c in seq if len(c) > 0]) for seq in train_seq)
    # min_note = min(min([min(c) for c in seq if len(c) > 0]) for seq in train_seq)

    return (train_seq, max_seq_length, max_note)

def sequences_to_data(seqs, seq_length, num_notes):
    """ Args: list of sequences, seq length, number of possible notes
        Returns data and targets matrices

        Note: adds one "note" to the vocab, which represents the end of the sequence
    """

    # add 1 to dimension for "end of sequence" token
    input_dim = num_notes + 1
    # (sequence length) X (num sequences) X (dimension)
    data = np.zeros([seq_length, len(seqs), input_dim], dtype=np.int32)
    for idx, seq in enumerate(seqs):
      for c_idx, chord in enumerate(seq):
          data[c_idx, idx, chord] = 1

    # roll back the time steps axis to get the target of each example
    targets = np.roll(data, -1, axis=0)
    # set final time step to "end of sequence" token
    targets[-1, :, :] = 0
    targets[-1, :, input_dim-1] = 1

    # sanity check
    assert targets.shape[0] == data.shape[0]
    for i in range(targets.shape[0]-1):
        assert data[i+1, :, :].tolist() == targets[i, :, :].tolist()
    # print 'passed sanity checks'

    return (data, targets)

def load_data():
    train_seqs, train_len, train_max = prepare_data('train')
    valid_seqs, valid_len, valid_max = prepare_data('valid')
    test_seqs, test_len, test_max = prepare_data('test')

    # num_notes is the maximum note value + 1 (0 is a possible note value)
    num_notes = max(train_max, valid_max, test_max)+1

    train_data, train_targets = sequences_to_data(train_seqs, train_len, num_notes)
    test_data, test_targets = sequences_to_data(test_seqs, test_len, num_notes)
    valid_data, valid_targets = sequences_to_data(valid_seqs, valid_len, num_notes)

    # input dim is number of notes plus end of sequence token
    input_dim = num_notes + 1

    return {
        "train": {
            "data": train_data,
            "targets": train_targets,
            "seq_length": train_len
        },
        "test": {
            "data": test_data,
            "targets": test_targets,
            "seq_length": test_len 
        },
        "valid": {
            "data": valid_data,
            "targets": valid_targets,
            "seq_length": valid_len
        },
        "input_dim": input_dim
    }
