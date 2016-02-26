import os
import math
import numpy as np
import tensorflow as tf    
import cPickle

import midi_util

def prepare_targets(data):
    # roll back the time steps axis to get the target of each example
    targets = np.roll(data, -1, axis=0)
    # set final time step to "end of sequence" token
    targets[-1, :, :] = 0

    # sanity check
    assert targets.shape[0] == data.shape[0]
    for i in range(targets.shape[0]-1):
        assert data[i+1, :, :].tolist() == targets[i, :, :].tolist()
    # print 'passed sanity checks'

    return targets

def parse_midi_directory(input_dir, time_step):
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [ \
        midi_util.parse_midi_to_sequence(f, time_step=time_step) \
        for f in files ]

    return sequences

def batch_data(sequences, time_batch_len=-1, max_time_batches=-1, verbose=False):
    """
    time_step: dataset-specific time step the MIDI should be broken up into (see parse_midi_to_sequence
               for more details
    time_batch_len: the max unrolling that will take place over BPTT. If -1 then set equal to the length
                    of the longest sequence.
    max_time_batches: the maximum amount of time batches. Every sequence greater than max_time_batches * 
                      time_batch_len is thrown away. If -1, there is no max amount of time batches
    """

    dims = sequences[0].shape[1]
    sequence_lens = [s.shape[0] for s in sequences]
    longest_seq = max(sequence_lens)

    if verbose:
        avg_seq_len = sum(sequence_lens) / len(sequences)
        print "Average Sequence Length: {}".format(avg_seq_len)
        print "Max Sequence Length: {}".format(time_batch_len)
        print "Number of sequences: {}".format(len(sequences))

    if time_batch_len < 0:
        time_batch_len = longest_seq

    total_time_batches = int(math.ceil(float(longest_seq)/float(time_batch_len)))

    if max_time_batches >= 0:
        num_time_batches = min(total_time_batches, max_time_batches)
        max_len = time_batch_len * num_time_batches
        sequences = filter(lambda x: len(x) <= max_len, sequences)
        sequence_lens = [s.shape[0] for s in sequences]
    else:
        num_time_batches = total_time_batches

    if verbose:
        print "Number of time batches: {}".format(num_time_batches)
        print "Number of sequences after filtering: {}".format(len(sequences))

    unsplit = list()
    unrolled_lengths = list()
    for sequence in sequences:
        unrolled_lengths.append(sequence.shape[0])
        copy = sequence.copy()
        copy.resize((time_batch_len * num_time_batches, dims)) 
        unsplit.append(copy)

    stacked = np.dstack(unsplit)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
    all_batches = np.swapaxes(stacked, 1, 2)
    all_targets = prepare_targets(all_batches)

    # sanity checks
    assert all_batches.shape == all_targets.shape
    assert all_batches.shape[1] == len(sequences)
    assert all_batches.shape[2] == dims

    batches = np.split(all_batches, [j * time_batch_len for j in range(1, num_time_batches)], axis=0)
    targets = np.split(all_targets, [j * time_batch_len for j in range(1, num_time_batches)], axis=0)

    assert len(batches) == len(targets) == num_time_batches

    rolled_lengths = [list() for i in range(num_time_batches)]
    for length in unrolled_lengths: 
        for time_step in range(num_time_batches): 
            step = time_step * time_batch_len
            if length <= step:
                rolled_lengths[time_step].append(0)
            else:
                rolled_lengths[time_step].append(min(time_batch_len, length - step))

    # time_batches = [list() for n in range(num_time_batches)]
    # time_batches_lens = [list() for n in range(num_time_batches)]
    # for sequence in sequences:
    #     batches = np.split(sequence, [j * time_batch_len for j in range(1, num_time_batches)])
    #     assert len(batches) == len(time_batches) == num_time_batches
    #     for t, batch in enumerate(batches):
    #         copy = batch.copy()
    #         batch_len = copy.shape[0]
    #         copy.resize((time_batch_len, dims))
    #         time_batches[t].append(copy)
    #         time_batches_lens[t].append(batch_len)
    #
    # data = []
    # for batch in time_batches:
    #     stacked = np.dstack(batch)
    #     swapped = np.swapaxes(stacked, 1, 2)
    #     data.append(swapped)

    # return batch, np.array(sequence_lengths)
    return batches, targets, rolled_lengths, unrolled_lengths, 

def load_data(data_dir, time_step, time_batch_len, max_time_batches, nottingham=False):

    data = {}

    if nottingham:
        with open(data_dir, 'r') as f:
            pickle = cPickle.load(f)

    for dataset in ['train', 'test', 'valid']:

        if nottingham:
            sequences = pickle[dataset]
        else:
            sequences = parse_midi_directory(os.path.join(data_dir, dataset), time_step)

        if dataset == 'test':
            mtb = -1
        else:
            mtb = max_time_batches

        notes, targets, seq_lengths, unrolled_lengths = batch_data(sequences, time_batch_len, mtb)

        data[dataset] = {
            "data": notes,
            "targets": targets,
            "seq_lengths": seq_lengths,
            "unrolled_lengths": unrolled_lengths,
            "time_batch_len": time_batch_len
        }

        data["input_dim"] = notes[0].shape[2]

    return data

def run_epoch(session, model, data, training=False):

    # change each data into a batch of data if it isn't already
    for n in ["data", "targets", "seq_lengths"]:
        if not isinstance(data[n], list):
            data[n] = [ data[n] ]

    target_tensors = [model.loss, model.final_state]
    if training:
        target_tensors.append(model.train_step)

    state = model.initial_state.eval()
    loss = 0
    for t in range(len(data["data"])):
        results = session.run(
            target_tensors,
            feed_dict={
                model.initial_state: state,
                model.seq_input: data["data"][t],
                model.seq_targets: data["targets"][t],
                model.seq_input_lengths: data["seq_lengths"][t],
                model.unrolled_lengths: data["unrolled_lengths"]
            })

        loss += results[0]
        state = results[1]

    return loss
