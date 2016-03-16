import os, sys
import argparse
import time
 
import cPickle
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import nottingham_util
import sampling
import util
from model import Model, NottinghamModel

###############################################################################
# TODO:
#   1. shuffle the batches randomly? 
#   2. add batches by size!
###############################################################################

def make_model_name(layers, units, melody_coeff=None):
    if melody_coeff:
        model_name = "nl_" + str(layers) + \
                     "_hs_" + str(units) + \
                     "_co_" + str(melody_coeff).replace(".", "p")
    else:
        model_name = "nl_" + str(layers) + \
                     "_hs_" + str(units)

    return model_name

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=float, default=0.10,
        help="Relative increase over lowest validation error required for early stopping")

    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--sample_seq', type=str, default='random',
        choices = ['random', 'chords'])
    parser.add_argument('--conditioning', type=int, default=-1)

    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--charts_dir', type=str, default='charts')
    parser.add_argument('--dataset', type=str, default='bach',
                        choices = ['bach', 'nottingham'])

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--melody_coeff', type=float, default=-1)

    args = parser.parse_args()

    learning_rate = 5e-3
    learning_rate_decay = 0.9

    if args.softmax:
        # learning_rate = 1e-2
        resolution = 480
        time_step = 120
        time_batch_len = 128
        num_time_batches = 3
        batch_size = 100

        with open(nottingham_util.PICKLE_LOC, 'r') as f:
            pickle = cPickle.load(f)
            chord_to_idx = pickle['chord_to_idx']

        data = util.load_data('', time_step, 
            time_batch_len, num_time_batches, nottingham=pickle)
        model_class = NottinghamModel

        model_suffix = '_softmax.model'
        charts_suffix = '_softmax.png'
    else:
        if args.dataset == 'bach':
            # learning_rate = 1e-2
            raise Exception("TODO: define stuff")
            data_dir = 'data/JSBChorales'
            resolution = 100
            time_step = 120
            time_batch_len = 100
            max_time_batches = -1
            batch_size = 100

        elif args.dataset == 'nottingham':
            data_dir = 'data/Nottingham'
            resolution = 480
            time_step = 120
            time_batch_len = 128
            num_time_batches = 3
            batch_size = 100

        else:
            raise Exception("unrecognized dataset")

        data = util.load_data(data_dir, time_step, time_batch_len, max_time_batches)
        model_class = Model

        model_suffix = '_' + args.dataset + '.model'
        charts_suffix = '_' + args.dataset + '.png'

    input_dim = data["input_dim"]

    print 'Finished loading data, input dim: {}'.format(input_dim)

    default_config = {
        "input_dim": input_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "time_batch_len": time_batch_len,
        "dropout_prob": 0.5,
        "cell_type": "lstm",
    } 

    def set_config(config, name):
        return dict(config, **{
            "time_batch_len": data[name]["data"][0].shape[0]
        })

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    if args.train:
        best_config = None
        best_valid_loss = None
        best_model_name = None

        # for melody_coeff in [0.5, 0.0, 1.0, 0.25, 0.75]:
        for melody_coeff in [0.5]:
            for num_layers in [1, 2, 3]:
                for hidden_size in [100, 150, 200]:

                    # model_name = make_model_name(num_layers, hidden_size, melody_coeff)
                    model_name = make_model_name(num_layers, hidden_size)

                    config = dict(default_config, **{
                        "input_dim": input_dim,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                    })

                    with tf.Graph().as_default(), tf.Session() as session:
                        with tf.variable_scope(model_name, reuse=None):
                            train_model = model_class(set_config(config, "train"), 
                                                      training=True)
                        with tf.variable_scope(model_name, reuse=True):
                            valid_model = model_class(set_config(config, "valid"))

                        saver = tf.train.Saver(tf.all_variables())
                        tf.initialize_all_variables().run()

                        train_model.assign_lr(session, learning_rate)
                        train_model.assign_lr_decay(session, learning_rate_decay)
                        if args.softmax:
                            train_model.assign_melody_coeff(session, melody_coeff)
                            valid_model.assign_melody_coeff(session, melody_coeff)

                        # training
                        early_stop_best_loss = None
                        start_saving = False
                        saved_flag = False
                        train_losses, valid_losses = [], []
                        start_time = time.time()
                        for i in range(args.num_epochs):
                            loss = util.run_epoch(session, train_model, 
                                data["train"], training=True, batch_size = batch_size)
                            train_losses.append((i, loss))
                            if i == 0:
                                continue

                            print 'Epoch: {}, Train Loss: {}, Time Per Epoch: {}'.format(i, loss, (time.time() - start_time)/i)
                            valid_loss = util.run_epoch(session, valid_model, data["valid"], training=False, batch_size = batch_size)
                            valid_losses.append((i, valid_loss))
                            print 'Valid Loss: {}'.format(valid_loss)
                            # if it's best validation loss so far, save it
                            if early_stop_best_loss == None:
                                early_stop_best_loss = valid_loss
                            elif valid_loss < early_stop_best_loss:
                                early_stop_best_loss = valid_loss
                                if start_saving:
                                    print 'Best loss so far encountered, saving model.'
                                    saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))
                                    saved_flag = True
                            elif not start_saving:
                                start_saving = True 
                                print 'Valid loss increased for the first time, will start saving models'
                                saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))
                                saved_flag = True

                            # early stop if generalization loss is worst than args.early_stopping
                            if args.early_stopping > 0:
                                if ((valid_loss / early_stop_best_loss) - 1.0) > args.early_stopping:
                                    print 'Early stopping criteria reached: {}'.format(args.early_stopping)
                                    break

                        if not saved_flag:
                            saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))

                        # set loss axis max to 20
                        axes = plt.gca()
                        if args.softmax:
                            axes.set_ylim([0, 2])
                        else:
                            axes.set_ylim([0, 100])
                        plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
                        plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
                        plt.legend(['Train Loss', 'Validation Loss'])
                        plt.savefig(os.path.join(args.charts_dir, model_name + charts_suffix))
                        plt.clf()
                        # print "Saved graph"

                        print "Model {}, Loss: {}".format(model_name, early_stop_best_loss)
                        if best_valid_loss == None or early_stop_best_loss < best_valid_loss:
                            print "Found best new model: {}".format(model_name)
                            best_valid_loss = early_stop_best_loss
                            best_config = config
                            best_model_name = model_name

        print 'Best config ({}): {}'.format(best_model_name, best_config)
        sample_model_name = best_model_name

    else:
        sample_model_name = make_model_name(args.num_layers, args.hidden_size, 
            args.melody_coeff if args.melody_coeff > 0 else None)

    # # SAMPLING SESSION #

    if not args.test and not args.sample:
        sys.exit(0)

    with tf.Graph().as_default(), tf.Session() as session:

        if args.sample: 
            with tf.variable_scope(sample_model_name, reuse=None):
                sampling_model = model_class(dict(default_config, **{
                    "time_batch_len": 1
                }))

        if args.test:
            test_config = set_config(default_config, "test")
            with tf.variable_scope(sample_model_name, reuse=True if args.sample else None):
                test_model = model_class(test_config)

        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(args.model_dir, sample_model_name + model_suffix)
        saver.restore(session, model_path)

        # Deterministic Testing
        if args.test: 
            if args.softmax and args.melody_coeff > 0:
                test_model.assign_melody_coeff(session, args.melody_coeff)
                print "Using melody_coeff: {}".format(test_model.melody_coeff.eval())
            test_loss, test_probs = util.run_epoch(session, test_model, data["test"], training=False, testing=True)

            print 'Testing Loss ({}): {}'.format(sample_model_name, test_loss)

            if args.softmax:
                nottingham_util.accuracy(test_probs, data['test']['targets'], test_config)
            else:
                util.accuracy(test_probs, data['test']['targets'], test_config, num_samples=50)

        # start with the first chord
        if args.sample:
            state = sampling_model.get_cell_zero_state(session, 1)
            sampling_length = time_batch_len * num_time_batches

            if args.sample_seq == 'chords':
                # 16 - one measure, 64 - chord progression
                repeats = (time_batch_len * num_time_batches) / 64
                sample_seq = nottingham_util.i_vi_iv_v(chord_to_idx, repeats, input_dim)
                print 'Sampling melody using a I, VI, IV, V progression'

            if args.sample_seq == 'random':
                sample_index = np.random.choice(np.arange(0, data["test"]["data"][0].shape[1]))
                sample_seq = [data["test"]["data"][i/time_batch_len][i%time_batch_len, sample_index, :] for
                    i in range(sampling_length)]    
                print "Sampling File: {} ({} time steps)".format(
                    data["test"]["metadata"][sample_index]['name'], len(sample_seq))

            chord = sample_seq[0]
            seq = [chord]

            if args.conditioning > 0:
                for i in range(1, args.conditioning):
                    seq_input = np.reshape(chord, [1, 1, input_dim])
                    feed = {
                        sampling_model.seq_input: seq_input,
                        sampling_model.initial_state: state,
                    }
                    state = session.run(sampling_model.final_state, feed_dict=feed)
                    chord = sample_seq[i]
                    seq.append(chord)

            if args.softmax:
                writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
                sampler = nottingham_util.NottinghamSampler(chord_to_idx, verbose=False)
            else:
                writer = midi_util.MidiWriter()
                sampler = sampling.Sampler(verbose=False)

            for i in range(max(sampling_length - len(seq), 0)):
                seq_input = np.reshape(chord, [1, 1, input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                [probs, state] = session.run(
                    [sampling_model.probs, sampling_model.final_state],
                    feed_dict=feed)
                probs = np.reshape(probs, [input_dim])
                chord = sampler.sample_notes(probs)

                if args.softmax:
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
