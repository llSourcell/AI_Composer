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
#   1. linear search over melody coeffecients
#   2. add dropout to the softmax layer?
###############################################################################

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--softmax', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay', type=float, default=0.90)
    parser.add_argument('--early_stopping', type=float, default=0.05,
        help="Relative increase over lowest validation error required for early stopping")
    parser.add_argument('--sample_length', type=int, default=200)
    parser.add_argument('--sample_melody', action='store_true', default=False)
    parser.add_argument('--sample_harmony', action='store_true', default=False)
    parser.add_argument('--conditioning', type=int, default=-1)

    parser.add_argument('--model_name', type=str, default='default_model')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--charts_dir', type=str, default='charts')

    parser.add_argument('--dataset', type=str, default='bach',
                        choices = ['bach', 'nottingham'])

    args = parser.parse_args()

    if args.softmax:
        resolution = 480
        time_step = 120
        time_batch_len = 100
        max_time_batches = 10

        with open(nottingham_util.PICKLE_LOC, 'r') as f:
            pickle = cPickle.load(f)
            chord_to_idx = pickle['chord_to_idx']

        data = util.load_data('', time_step, 
            time_batch_len, max_time_batches, nottingham=pickle)
        model_class = NottinghamModel

        #TODO: change to distinguish from regular nottingham model
        model_suffix = '_softmax.model'
        charts_suffix = '_softmax.png'
    else:
        if args.dataset == 'bach':
            data_dir = 'data/JSBChorales'
            resolution = 100
            time_step = 120
            time_batch_len = 100 # use the longest seq length
            max_time_batches = -1 
        elif args.dataset == 'nottingham':
            data_dir = 'data/Nottingham'
            resolution = 480
            time_step = 120
            time_batch_len = 100
            max_time_batches = 10
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
        "hidden_size": 100,
        "num_layers": 2,
        "dropout_prob": 0.5,
        "cell_type": "lstm",
    } 

    def set_config(config, name):
        return dict(config, **{
            "batch_size": data[name]["data"][0].shape[1],
            "time_batch_len": data[name]["data"][0].shape[0]
        })

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    if args.train:
        best_config = None
        best_valid_loss = None
        best_model_name = None

        for melody_coeff in [0.5]:
            for num_layers in [1, 2]:
                for hidden_size in [50, 100, 150]:
                    model_name = "nl_" + str(num_layers) + \
                                 "_hs_" + str(hidden_size)
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

                        train_model.assign_lr(session, args.learning_rate)
                        train_model.assign_lr_decay(session, args.learning_rate_decay)
                        train_model.assign_melody_coeff(session, melody_coeff)

                        # training
                        early_stop_best_loss = None
                        start_saving = False
                        saved_flag = False
                        train_losses, valid_losses = [], []
                        start_time = time.time()
                        for i in range(args.num_epochs):
                            loss = util.run_epoch(session, train_model, 
                                data["train"], training=True)
                            train_losses.append((i, loss))
                            if i % 10 == 0 and i != 0:
                                valid_loss = util.run_epoch(session, valid_model, data["valid"])
                                valid_losses.append((i, valid_loss))
                                print 'Epoch: {}, Train Loss: {}, Valid Loss: {}, Time Per Epoch: {}'.format(
                                    i, loss, valid_loss, (time.time() - start_time)/i)

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

                                # early stop if generalization loss is worst than args.early_stopping
                                if args.early_stopping > 0:
                                    if ((valid_loss / early_stop_best_loss) - 1.0) > args.early_stopping:
                                        print 'Early stopping criteria reached: {}'.format(args.early_stopping)
                                        break

                        if not saved_flag:
                            saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))

                        # set loss axis max to 20
                        axes = plt.gca()
                        axes.set_ylim([0, 20])
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
        sample_model_name = args.model_name


    # # SAMPLING SESSION #

    do_sampling = args.sample_length > 0
    if not args.test and not do_sampling:
        sys.exit(0)

    with tf.Graph().as_default(), tf.Session() as session:

        if do_sampling: 
            with tf.variable_scope(sample_model_name, reuse=None):
                sampling_model = model_class(dict(default_config, **{
                    "batch_size": 1,
                    "time_batch_len": 1
                }))

        if args.test:
            test_config = set_config(default_config, "test")
            with tf.variable_scope(sample_model_name, reuse=True if do_sampling else None):
                test_model = model_class(test_config)

        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(args.model_dir, sample_model_name + model_suffix)
        saver.restore(session, model_path)

        # Deterministic Testing
        if args.test: 
            test_loss, test_probs = util.run_epoch(session, test_model, 
                                                   data["test"], 
                                                   training=False, testing=True)

            print 'Testing Loss ({}): {}'.format(sample_model_name, test_loss)

            if args.softmax:
                nottingham_util.accuracy(test_probs, pickle['test'], test_config)
            else:
                util.accuracy(test_probs, data['test']['targets'], 
                    data['test']['unrolled_lengths'], test_config, num_samples=50)

        # start with the first chord
        if do_sampling:
            state = sampling_model.initial_state.eval()
            sample_index = np.random.choice(np.arange(0, data["test"]["data"][0].shape[1]))
            sampling_length = max(data["test"]["unrolled_lengths"][sample_index], args.sample_length)
            print "Sampling File: {} ({} time steps)".format(
                data["test"]["metadata"][sample_index]['name'], sampling_length)

            chord = data["test"]["data"][0][0, sample_index, :]

            # if args.softmax:
            #     chord_idx = nottingham_util.NOTTINGHAM_MELODY_RANGE + chord_to_idx["CM"]
            #     melody_idx = 24
            #     chord = np.zeros(input_dim)
            #     chord[chord_idx] = 1
            #     chord[melody_idx] = 1

            seq = [chord]

            if args.conditioning > 0:
                for i in range(1, args.conditioning):
                    seq_input = np.reshape(chord, [1, 1, input_dim])
                    feed = {
                        sampling_model.seq_input: seq_input,
                        sampling_model.initial_state: state,
                        sampling_model.seq_input_lengths: [1]
                    }
                    state = session.run(sampling_model.final_state, feed_dict=feed)
                    chord = data["test"]["data"][i/time_batch_len][i%time_batch_len, sample_index, :]
                    seq.append(chord)

            if args.softmax:
                writer = nottingham_util.NottinghamMidiWriter(chord_to_idx, verbose=True)
                sampler = nottingham_util.NottinghamSampler(chord_to_idx, verbose=True)
            else:
                writer = midi_util.MidiWriter()
                sampler = sampling.Sampler(min_prob = args.temp, verbose=False)

            for i in range(max(sampling_length - len(seq), 0)):
                seq_input = np.reshape(chord, [1, 1, input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                    sampling_model.seq_input_lengths: [1]
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
                        chord[r:] = data["test"]["data"][i/time_batch_len][i%time_batch_len, sample_index, :][r:]
                    elif args.sample_harmony:
                        chord[:r] = 0
                        chord[:r] = data["test"]["data"][i/time_batch_len][i%time_batch_len, sample_index, :][:r]

                seq.append(chord)

            writer.dump_sequence_to_midi(seq, "best.midi", 
                time_step=time_step, resolution=resolution)
