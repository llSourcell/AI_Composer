import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import midi_util
import sampling
import util
from model import Model

###############################################################################
# TODO:
# 1. figure out a way to train different sequence lengths
# 2. change test/accuracy code to reflect loss over the entire sequence instead
#    of a fixed length of it
###############################################################################

if __name__ == '__main__':
    np.random.seed(1)      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--learning_rate_decay', type=float, default=0.90)
    parser.add_argument('--early_stopping', type=float, default=0.15, 
        help="Relative increase over lowest validation error required for early stopping")
    parser.add_argument('--sample_length', type=int, default=200)
    parser.add_argument('--conditioning', type=int, default=-1)

    parser.add_argument('--model_name', type=str, default='default_model')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--charts_dir', type=str, default='charts')

    parser.add_argument('--dataset', type=str, default='bach',
                        choices = ['bach', 'nottingham'])

    args = parser.parse_args()

    if args.dataset == 'bach':
        data_dir = 'data/JSBChorales'
        resolution = 100
        time_step = 120
        time_batch_len = -1 # use the longest seq length
        max_time_batches = -1 # unroll to longest seq
    elif args.dataset == 'nottingham':
        data_dir = 'data/Nottingham'
        resolution = 480
        time_step = 120 
        # TODO: tweak below
        time_batch_len = 100
        max_time_batches = -1 # use as many time batches as needed
    else:
        raise Exception("unrecognized dataset")

    model_suffix = '_' + args.dataset + '.model'
    charts_suffix = '_' + args.dataset + '.png'

    data = util.load_data(data_dir, time_step, time_batch_len, max_time_batches)
    input_dim = data["input_dim"]
    print 'Finished loading data, input dim: {}'.format(input_dim)

    default_config = {
        "input_dim": input_dim,
        "hidden_size": 100,
        "num_layers": 1,
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

        for num_layers in [1]:
            for hidden_size in [100]:
                for learning_rate in [1e-2]:

                    model_name = "nl_" + str(num_layers) + \
                                 "_hs_" + str(hidden_size) + \
                                 "_lr_" + str(learning_rate).replace(".", "p")
                    config = dict(default_config, **{
                        "input_dim": input_dim,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                    })

                    with tf.Graph().as_default(), tf.Session() as session:
                        with tf.variable_scope(model_name, reuse=None):
                            train_model = Model(set_config(config, "train"), 
                                                training=True)
                        with tf.variable_scope(model_name, reuse=True):
                            valid_model = Model(set_config(config, "valid"))

                        saver = tf.train.Saver(tf.all_variables())
                        tf.initialize_all_variables().run()

                        # training
                        early_stop_best_loss = None
                        train_losses, valid_losses = [], []
                        train_model.assign_lr(session, learning_rate)
                        train_model.assign_lr_decay(session, args.learning_rate_decay)
                        for i in range(args.num_epochs):
                            loss = util.run_epoch(session, train_model, 
                                data["train"], training=True)
                            train_losses.append((i, loss))
                            if i % 10 == 0:
                                valid_loss = util.run_epoch(session, valid_model, data["valid"])
                                valid_losses.append((i, valid_loss))
                                print 'Epoch: {}, Train Loss: {}, Valid Loss: {}'.format(i, loss, valid_loss)

                                # early stop if generalization loss is worst than args.early_stopping
                                early_stop_best_loss = min(early_stop_best_loss, valid_loss) if early_stop_best_loss != None else valid_loss
                                if ((valid_loss / early_stop_best_loss) - 1.0) > args.early_stopping:
                                    print 'Early stopping criteria reached: {}'.format(args.early_stopping)
                                    break

                        saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))
                        # print "Saved model"

                        # set loss axis max to 20
                        axes = plt.gca()
                        axes.set_ylim([0, 20])
                        plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
                        plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
                        plt.legend(['Train Loss', 'Validation Loss'])
                        plt.savefig(os.path.join(args.charts_dir, model_name + charts_suffix))
                        plt.clf()
                        # print "Saved graph"

                        valid_loss = util.run_epoch(session, valid_model, data["valid"])
                        print "Model {} Loss: {}".format(model_name, valid_loss)
                        if best_valid_loss == None or valid_loss < best_valid_loss:
                            print "Found best new model: {}".format(model_name)
                            best_valid_loss = valid_loss
                            best_config = config
                            best_model_name = model_name

        print 'Best config ({}): {}'.format(best_model_name, best_config)
        sample_model_name = best_model_name

    else:
        sample_model_name = args.model_name


    # # SAMPLING SESSION #

    with tf.Graph().as_default(), tf.Session() as session:

        with tf.variable_scope(sample_model_name, reuse=None):
            sampling_model = Model(dict(default_config, **{
                "batch_size": 1,
                "time_batch_len": 1
            }))
        with tf.variable_scope(sample_model_name, reuse=True):
            test_model = Model(set_config(default_config, "test"))

        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(args.model_dir, sample_model_name + model_suffix)
        saver.restore(session, model_path)

        # Deterministic Testing
        test_loss = util.run_epoch(session, test_model, data["test"], training=False)
        print 'Testing Loss ({}): {}'.format(sample_model_name, test_loss)

        # TODO: rewrite
        # predicted = (test_probs > 0.5).astype(np.float32)
        #
        # true_positives = np.sum(np.multiply(predicted == test_targets, predicted))
        # false_positives = np.sum(np.multiply(predicted != test_targets, predicted))
        # false_negatives = np.sum(np.multiply(predicted != test_targets, test_targets))
        #
        # total_predicted = np.sum(np.multiply(predicted, predicted))
        # total_targets = np.sum(np.multiply(test_targets, test_targets))
        # if total_predicted != 0 and total_targets != 0:
        #     precision = float(true_positives) / float(total_predicted)
        #     recall = float(true_positives) / float(total_targets)
        #     print 'Precision: {}'.format(precision)
        #     print 'Recall: {}'.format(recall)
        #     print 'F1 Score: {}'.format(2 * (precision * recall) / (precision + recall))
        #     accuracy = float(true_positives) / float(true_positives + false_positives + false_negatives)
        #     print 'Accuracy: {}'.format(accuracy)
        # else:
        #     print 'Total predicted and/or total targets == 0, there may be an error'


        # start with the first chord
        # chord = midi_util.cmaj()
        state = sampling_model.initial_state.eval()
        sampler = sampling.Sampler(min_prob = args.temp, verbose=True)
        sampling_length = 40

        chord = data["train"]["data"][0][0, 0, :]
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
                chord = data["train"]["data"][i, 0, :]
                seq.append(chord)

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
            # chord = sampler.sample_notes_prob(probs, max_notes=8)
            chord = sampler.sample_notes_static(probs)
            seq.append(chord)

        midi_util.dump_sequence_to_midi(seq, "best.midi", 
            time_step=time_step, resolution=resolution)
