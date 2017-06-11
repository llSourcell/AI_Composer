import os
import math
import cPickle
from collections import defaultdict
from random import shuffle

import numpy as np
import tensorflow as tf    

import midi_util
import nottingham_util

def parse_midi_directory(input_dir, time_step):
    """ 
    input_dir: data directory full of midi files
    time_step: the number of ticks to use as a time step for discretization

    Returns a list of [T x D] matrices, where T is the amount of time steps
    and D is the range of notes.
    """
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [ \
        (f, midi_util.parse_midi_to_sequence(f, time_step=time_step)) \
        for f in files ]

    return sequences

def batch_data(sequences, time_batch_len=128, max_time_batches=10,
               softmax=False, verbose=False):
    """
    sequences: a list of [T x D] matrices, each matrix representing a sequencey
    time_batch_len: the unrolling length that will be used by BPTT. 
    max_time_batches: the max amount of time batches to consider. Any sequences 
                      longert than max_time_batches * time_batch_len will be ignored
                      Can be set to -1 to all time batches needed.
    softmax: Flag should be set to true if using the dual-softmax formualtion

    returns [
        [ [ data ], [ target ] ], # batch with one time step
        [ [ data1, data2 ], [ target1, target2 ] ], # batch with two time steps
        ...
    ]
    """

    assert time_batch_len > 0

    dims = sequences[0].shape[1]
    sequence_lens = [s.shape[0] for s in sequences]

    if verbose:
        avg_seq_len = sum(sequence_lens) / len(sequences)
        print "Average Sequence Length: {}".format(avg_seq_len)
        print "Max Sequence Length: {}".format(time_batch_len)
        print "Number of sequences: {}".format(len(sequences))

    batches = defaultdict(list)
    for sequence in sequences:
        # -1 because we can't predict the first step
        num_time_steps = ((sequence.shape[0]-1) // time_batch_len) 
        if num_time_steps < 1:
            continue
        if max_time_batches > 0 and num_time_steps > max_time_batches:
            continue
        batches[num_time_steps].append(sequence)

    if verbose:
        print "Batch distribution:"
        print [(k, len(v)) for (k, v) in batches.iteritems()]

    def arrange_batch(sequences, num_time_steps):
        sequences = [s[:(num_time_steps*time_batch_len)+1, :] for s in sequences]
        stacked = np.dstack(sequences)
        # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
        data = np.swapaxes(stacked, 1, 2)
        targets = np.roll(data, -1, axis=0)
        # cutoff final time step
        data = data[:-1, :, :]
        targets = targets[:-1, :, :]
        assert data.shape == targets.shape

        if softmax:
            r = nottingham_util.NOTTINGHAM_MELODY_RANGE
            labels = np.ones((targets.shape[0], targets.shape[1], 2), dtype=np.int32)
            assert np.all(np.sum(targets[:, :, :r], axis=2) == 1)
            assert np.all(np.sum(targets[:, :, r:], axis=2) == 1)
            labels[:, :, 0] = np.argmax(targets[:, :, :r], axis=2)
            labels[:, :, 1] = np.argmax(targets[:, :, r:], axis=2)
            targets = labels
            assert targets.shape[:2] == data.shape[:2]

        assert data.shape[0] == num_time_steps * time_batch_len

        # split them up into time batches
        tb_data = np.split(data, num_time_steps, axis=0)
        tb_targets = np.split(targets, num_time_steps, axis=0)

        assert len(tb_data) == len(tb_targets) == num_time_steps
        for i in range(len(tb_data)):
            assert tb_data[i].shape[0] == time_batch_len
            assert tb_targets[i].shape[0] == time_batch_len
            if softmax:
                assert np.all(np.sum(tb_data[i], axis=2) == 2)

        return (tb_data, tb_targets)

    return [ arrange_batch(b, n) for n, b in batches.iteritems() ]
        
def load_data(data_dir, time_step, time_batch_len, max_time_batches, nottingham=None):
    """
    nottingham: The sequences object as created in prepare_nottingham_pickle
                (see nottingham_util for more). If None, parse all the MIDI
                files from data_dir
    time_step: the time_step used to parse midi files (only used if data_dir
               is provided)
    time_batch_len and max_time_batches: see batch_data()

    returns { 
        "train": {
            "data": [ batch_data() ],
            "metadata: { ... }
        },
        "valid": { ... }
        "test": { ... }
    }
    """

    data = {}
    for dataset in ['train', 'test', 'valid']:

        # For testing, use ALL the sequences
        if dataset == 'test':
            max_time_batches = -1

        # Softmax formualation preparsed into sequences
        if nottingham:
            sequences = nottingham[dataset]
            metadata = nottingham[dataset + '_metadata']
        # Cross-entropy formulation needs to be parsed
        else:
            sf = parse_midi_directory(os.path.join(data_dir, dataset), time_step)
            sequences = [s[1] for s in sf]
            files = [s[0] for s in sf]
            metadata = [{
                'path': f,
                'name': f.split("/")[-1].split(".")[0]
            } for f in files]

        dataset_data = batch_data(sequences, time_batch_len, max_time_batches, softmax = True if nottingham else False)

        data[dataset] = {
            "data": dataset_data,
            "metadata": metadata,
        }

        data["input_dim"] = dataset_data[0][0][0].shape[2]

    return data


def run_epoch(session, model, batches, training=False, testing=False):
    """
    session: Tensorflow session object
    model: model object (see model.py)
    batches: data object loaded from util_data()

    training: A backpropagation iteration will be performed on the dataset
    if this flag is active

    returns average loss per time step over all batches.
    if testing flag is active: returns [ loss, probs ] where is the probability
        values for each note
    """

    # shuffle batches
    shuffle(batches)

    target_tensors = [model.loss, model.final_state]
    if testing:
        target_tensors.append(model.probs)
        batch_probs = defaultdict(list)
    if training:
        target_tensors.append(model.train_step)

    losses = []
    for data, targets in batches:
        # save state over unrolling time steps
        batch_size = data[0].shape[1]
        num_time_steps = len(data)
        state = model.get_cell_zero_state(session, batch_size) 
        probs = list()

        for tb_data, tb_targets in zip(data, targets):
            if testing:
                tbd = tb_data
                tbt = tb_targets
            else:
                # shuffle all the batches of input, state, and target
                batches = tb_data.shape[1]
                permutations = np.random.permutation(batches)
                tbd = np.zeros_like(tb_data)
                tbd[:, np.arange(batches), :] = tb_data[:, permutations, :]
                tbt = np.zeros_like(tb_targets)
                tbt[:, np.arange(batches), :] = tb_targets[:, permutations, :]
                state[np.arange(batches)] = state[permutations]

            feed_dict = {
                model.initial_state: state,
                model.seq_input: tbd,
                model.seq_targets: tbt,
            }
            results = session.run(target_tensors, feed_dict=feed_dict)

            losses.append(results[0])
            state = results[1]
            if testing:
                batch_probs[num_time_steps].append(results[2])

    loss = sum(losses) / len(losses)

    if testing:
        return [loss, batch_probs]
    else:
        return loss

def accuracy(batch_probs, data, num_samples=20):
    """
    batch_probs: probs object returned from run_epoch
    data: data object passed into run_epoch
    num_samples: the number of times to sample each note (an average over all
    these samples will be used)

    returns the accuracy metric according to
    http://ismir2009.ismir.net/proceedings/PS2-21.pdf
    """

    false_positives, false_negatives, true_positives = 0, 0, 0 
    for _, batch_targets in data:
        num_time_steps = len(batch_data)
        for ts_targets, ts_probs in zip(batch_targets, batch_probs[num_time_steps]):

            assert ts_targets.shape == ts_targets.shape

            for seq_idx in range(ts_targets.shape[1]):
                for step_idx in range(ts_targets.shape[0]):
                    for note_idx, prob in enumerate(ts_probs[step_idx, seq_idx, :]):
                        num_occurrences = np.random.binomial(num_samples, prob)
                        if ts_targets[step_idx, seq_idx, note_idx] == 0.0:
                            false_positives += num_occurrences
                        else:
                            false_negatives += (num_samples - num_occurrences)
                            true_positives += num_occurrences
                
    accuracy = (float(true_positives) / float(true_positives + false_positives + false_negatives)) 

    print "Precision: {}".format(float(true_positives) / (float(true_positives + false_positives)))
    print "Recall: {}".format(float(true_positives) / (float(true_positives + false_negatives)))
    print "Accuracy: {}".format(accuracy)
