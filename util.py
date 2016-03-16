import os
import math
import numpy as np
import tensorflow as tf    
import cPickle

import midi_util
import nottingham_util

def parse_midi_directory(input_dir, time_step):
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [ \
        (f, midi_util.parse_midi_to_sequence(f, time_step=time_step)) \
        for f in files ]

    return sequences

def batch_data(sequences, time_batch_len=128, num_time_batches=3,
               softmax=False, verbose=False):
    """
    time_step: dataset-specific time step the MIDI should be broken up into (see parse_midi_to_sequence
               for more details
    time_batch_len: the max unrolling that will take place over BPTT. If -1 then set equal to the length
                    of the longest sequence.
    num_time_batches: the amount of time batches; any sequences below this get thrown away
    """

    assert time_batch_len > 0
    assert num_time_batches > 0

    dims = sequences[0].shape[1]
    sequence_lens = [s.shape[0] for s in sequences]

    if verbose:
        avg_seq_len = sum(sequence_lens) / len(sequences)
        print "Average Sequence Length: {}".format(avg_seq_len)
        print "Max Sequence Length: {}".format(time_batch_len)
        print "Number of sequences: {}".format(len(sequences))

    # add one because we can't use the last time step as a target
    seq_length_cutoff = time_batch_len * num_time_batches + 1
    sequences = filter(lambda s: s.shape[0] >= seq_length_cutoff, sequences)

    if verbose:
        print "Number of sequences after filtering: {}".format(len(sequences))

    sequences = [s[:seq_length_cutoff, :] for s in sequences]
    stacked = np.dstack(sequences)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
    data = np.swapaxes(stacked, 1, 2)
    targets = np.roll(data, -1, axis=0)

    # cutoff final time step
    data = data[:-1, :, :]
    targets = targets[:-1, :, :]

    assert data.shape == targets.shape
    assert data.shape[0] == time_batch_len * num_time_batches
    assert data.shape[1] == len(sequences)
    assert data.shape[2] == dims

    if softmax:
        r = nottingham_util.NOTTINGHAM_MELODY_RANGE
        labels = np.ones((targets.shape[0], targets.shape[1], 2), dtype=np.int32)
        assert np.all(np.sum(targets[:, :, :r], axis=2) == 1)
        assert np.all(np.sum(targets[:, :, r:], axis=2) == 1)
        labels[:, :, 0] = np.argmax(targets[:, :, :r], axis=2)
        labels[:, :, 1] = np.argmax(targets[:, :, r:], axis=2)
        targets = labels
        assert targets.shape[:2] == data.shape[:2]

    batches = np.split(data, [j * time_batch_len for j in range(1, num_time_batches)], axis=0)
    targets = np.split(targets, [j * time_batch_len for j in range(1, num_time_batches)], axis=0)

    assert len(batches) == len(targets) == num_time_batches

    if softmax:
        for b, t in zip(batches, targets):
            assert np.all(np.sum(b, axis=2) == 2)

    return batches, targets

def load_data(data_dir, time_step, time_batch_len, num_time_batches, nottingham=None):

    data = {}

    if nottingham:
        pickle = nottingham

    for dataset in ['train', 'test', 'valid']:

        if nottingham:
            sequences = pickle[dataset]
            metadata = pickle[dataset + '_metadata']
        else:
            sf = parse_midi_directory(os.path.join(data_dir, dataset), time_step)
            sequences = [s[1] for s in sf]
            files = [s[0] for s in sf]
            metadata = [{
                'path': f,
                'name': f.split("/")[-1].split(".")[0]
            } for f in files]

        notes, targets = batch_data(sequences, time_batch_len, num_time_batches, \
            softmax = True if nottingham else False)

        data[dataset] = {
            "data": notes,
            "metadata": metadata,
            "targets": targets,
            "time_batch_len": time_batch_len
        }

        data["input_dim"] = notes[0].shape[2]

    return data

def run_epoch(session, model, data, training=False, testing=False, batch_size=-1, separate=False):

    # change each data into a batch of data if it isn't already
    for n in ["data", "targets"]:
        if not isinstance(data[n], list):
            data[n] = [ data[n] ]

    target_tensors = [model.loss, model.final_state]
    if testing:
        target_tensors.append(model.probs)
        prob_vals = None
    if training:
        target_tensors.append(model.train_step)

    total_examples = data["data"][0].shape[1]
    if batch_size < 0:
        batch_size = total_examples

    # 99/100 -> [0]
    # 100/100 -> [0]
    # 101/100 -> [0, 1]
    num_batches = int(math.ceil(float(total_examples)/batch_size))
    losses = []
    for batch in range(num_batches):
        probs = list()

        # HACK: handle special separate case:
        if separate:
            targets = [tb[:, (batch*batch_size):((batch+1)*batch_size)]
                        for tb in data["targets"]]
        else:
            targets = [tb[:, (batch*batch_size):((batch+1)*batch_size), :]
                        for tb in data["targets"]]

        batches = [tb[:, (batch*batch_size):((batch+1)*batch_size), :]
                    for tb in data["data"]]
        b_size = batches[0].shape[1]

        state = model.get_cell_zero_state(session, b_size)
        for t in range(len(batches)):
            feed_dict = {
                model.initial_state: state,
                model.seq_input: batches[t],
                model.seq_targets: targets[t],
            }
            results = session.run(target_tensors, feed_dict=feed_dict)

            losses.append(results[0])
            state = results[1]
            if testing:
                probs.append(results[2])
        
        if testing:
            if not prob_vals:
                prob_vals = probs
            else:
                prob_vals = [np.hstack((prob_vals[i], probs[i])) for i in range(len(prob_vals))]

    loss = sum(losses) / len(losses)

    if testing:
        return [loss, prob_vals]
    else:
        return loss

def accuracy(raw_probs, raw_targets, config, num_samples=20):
    
    time_batch_len = config["time_batch_len"]
    input_dim = config["input_dim"]

    # reshape probability batches into [time_batch_len * max_time_batches, batch_size, input_dim]
    test_probs = np.concatenate(raw_probs, axis=0)
    test_targets = np.concatenate(raw_targets, axis=0)

    false_positives, false_negatives, true_positives = 0, 0, 0 
    for seq_idx in range(test_targets.shape[1]):
        for step_idx in range(test_targets.shape[0]):

            # if we've reached the end of the sequence, go to next seq
            for note_idx, prob in enumerate(test_probs[step_idx, seq_idx, :]):  
                num_occurrences = np.random.binomial(num_samples, prob)
                if test_targets[step_idx, seq_idx, note_idx] == 0.0:
                    false_positives += num_occurrences
                else:
                    false_negatives += (num_samples - num_occurrences)
                    true_positives += num_occurrences
                
    accuracy = (float(true_positives) / float(true_positives + false_positives + false_negatives)) 

    print "Precision: {}".format(float(true_positives) / (float(true_positives + false_positives)))
    print "Recall: {}".format(float(true_positives) / (float(true_positives + false_negatives)))
    print "Accuracy: {}".format(accuracy)
