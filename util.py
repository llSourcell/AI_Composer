import os
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

def load_data(data_dir, min_time_step=-1, round_to=-1, max_seq_len=-1):

    training = midi_util.parse_midi_directory(os.path.join(data_dir, 'train'),
                                              min_time_step=min_time_step, round_to=round_to,
                                              max_seq_len=max_seq_len)
    testing = midi_util.parse_midi_directory(os.path.join(data_dir, 'train'),
                                             min_time_step=min_time_step, round_to=round_to,
                                             max_seq_len=max_seq_len)
    valid = midi_util.parse_midi_directory(os.path.join(data_dir, 'valid'),
                                           min_time_step=min_time_step, round_to=round_to,
                                           max_seq_len=max_seq_len)

    return {
        "train": {
            "data": training,
            "targets": prepare_targets(training),
            "seq_length": training.shape[1]
        },
        "test": {
            "data": testing,
            "targets": prepare_targets(testing),
            "seq_length": testing.shape[1]
        },
        "valid": {
            "data": valid,
            "targets": prepare_targets(valid),
            "seq_length": valid.shape[1]
        },
        "input_dim": training.shape[2]
    }

def run_epoch(session, model, seq_data, seq_targets, training=False):
    if training:
        _, loss_value = session.run(
            [model.train_step, model.loss],
            feed_dict={
                model.seq_input: seq_data,
                model.seq_targets: seq_targets
            })
    else:
        loss_value = session.run(
            model.loss,
            feed_dict={
                model.seq_input: seq_data,
                model.seq_targets: seq_targets
            })
    return loss_value
