import os
import math
import numpy as np
import tensorflow as tf    

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

def parse_midi_directory(input_dir, time_step, time_batch_len=-1, max_time_batches=-1, verbose=False):
    """
    time_step: dataset-specific time step the MIDI should be broken up into (see parse_midi_to_sequence
               for more details
    time_batch_len: the max unrolling that will take place over BPTT. If -1 then set equal to the length
                    of the longest sequence.
    max_time_batches: the maximum amount of time batches. Every sequence greater than max_time_batches * 
                      time_batch_len is thrown away. If -1, there is no max amount of time batches
    """
    # TODO: only uses the first time_batch_len... change this to incorporate a state-saving rnn somehow...

    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [ \
        midi_util.parse_midi_to_sequence(f, time_step=time_step) \
        for f in files ]
    dims = sequences[0].shape[1]

    longest_seq = max(s.shape[0] for s in sequences)

    if verbose:
        avg_seq_len = sum(s.shape[0] for s in sequences) / len(sequences)
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
    else:
        num_time_batches = total_time_batches

    if verbose:
        print "Number of time batches: {}".format(num_time_batches)
        print "Number of sequences after filtering: {}".format(len(sequences))

    time_batches = [list() for n in range(num_time_batches)]
    time_batches_lens = [list() for n in range(num_time_batches)]
    for sequence in sequences:
        batches = np.split(sequence, [j * time_batch_len for j in range(1, num_time_batches)])
        assert len(batches) == len(time_batches) == num_time_batches
        for t, batch in enumerate(batches):
            copy = batch.copy()
            batch_len = copy.shape[0]
            copy.resize((time_batch_len, dims))
            time_batches[t].append(copy)
            time_batches_lens[t].append(batch_len)

    data = []
    for batch in time_batches:
        stacked = np.dstack(batch)
        # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
        swapped = np.swapaxes(stacked, 1, 2)
        data.append(swapped)

    # return batch, np.array(sequence_lengths)
    return data, time_batches_lens

def load_data(data_dir, time_step, time_batch_len, max_time_batches):

    data = {}

    for dataset in ['train', 'test', 'valid']:
        dataset_data, dataset_lens = \
            parse_midi_directory(os.path.join(data_dir, dataset),
                time_step, time_batch_len)

        unrolled_seq_lengths = list()
        for s_idx in range(len(dataset_lens[0])):
            s_len = sum(dataset_lens[t][s_idx] for t in range(len(dataset_lens)))
            unrolled_seq_lengths.append(float(s_len))

        data[dataset] = {
            "data": dataset_data,
            "targets": [prepare_targets(d) for d in dataset_data],
            "time_batch_len": time_batch_len,
            "seq_lengths": dataset_lens,
            "unrolled_lengths": unrolled_seq_lengths
        }

        data["input_dim"] = dataset_data[0].shape[2]

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
