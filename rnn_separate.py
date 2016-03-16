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
from model import Model, NottinghamSeparate

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=float, default=0.10,
        help="Relative increase over lowest validation error required for early stopping")

    parser.add_argument('--choice', type=str, default='melody',
                        choices = ['melody', 'harmony'])

    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--charts_dir', type=str, default='charts')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=100)

    args = parser.parse_args()

    def make_model_name(layers, units):
        if args.choice == 'melody':
            model_name = "nl_" + str(layers) + \
                         "_hs_" + str(units) + \
                         "_melody"
        else:
            model_name = "nl_" + str(layers) + \
                         "_hs_" + str(units) + \
                         "_harmony"

        return model_name

    learning_rate = 1e-3
    learning_rate_decay = 0.9

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

    # cut away unnecessary parts
    r = nottingham_util.NOTTINGHAM_MELODY_RANGE
    if args.choice == 'melody':
        print "Using only melody"
        for d in ['train', 'test', 'valid']:
            data[d]["data"] = [ time_batch[:, :, :r] for time_batch in data[d]["data"] ]
            data[d]["targets"] = [ time_batch[:, :, 0] for time_batch in data[d]["targets"] ]
            # print data[d]["targets"][0].shape
    else:
        print "Using only harmony"
        for d in ['train', 'test', 'valid']:
            data[d]["data"] = [ time_batch[:, :, r:] for time_batch in data[d]["data"] ]
            data[d]["targets"] = [ time_batch[:, :, 1] for time_batch in data[d]["targets"] ]

    input_dim = data["input_dim"] = data['train']["data"][0].shape[2]
    print "New input dim: {}".format(input_dim)
    for i, tb in enumerate(data[d]["data"]):
        assert np.all(np.less_equal(np.sum(tb, axis=2), 1.0))


    model_class = NottinghamSeparate
    model_suffix = '_separate.model'
    charts_suffix = '_separate.png'

    print 'Finished loading data, input dim: {}'.format(input_dim)

    default_config = {
        "choice": args.choice,
        "input_dim": input_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "time_batch_len": time_batch_len,
        "dropout_prob": 0.5,
        "cell_type": "lstm",
    } 

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    model_name = make_model_name(args.num_layers, args.hidden_size)

    if args.train:

        with tf.Graph().as_default(), tf.Session() as session:
            with tf.variable_scope(model_name, reuse=None):
                train_model = model_class(default_config, training=True)
            with tf.variable_scope(model_name, reuse=True):
                valid_model = model_class(default_config)

            saver = tf.train.Saver(tf.all_variables())
            tf.initialize_all_variables().run()

            train_model.assign_lr(session, learning_rate)
            train_model.assign_lr_decay(session, learning_rate_decay)

            # training
            early_stop_best_loss = None
            start_saving = False
            saved_flag = False
            train_losses, valid_losses = [], []
            start_time = time.time()
            for i in range(args.num_epochs):
                loss = util.run_epoch(session, train_model, 
                    data["train"], training=True, batch_size = batch_size, separate=True)
                train_losses.append((i, loss))
                if i == 0:
                    continue

                print 'Epoch: {}, Train Loss: {}, Time Per Epoch: {}'.format(i, loss, (time.time() - start_time)/i)
                valid_loss = util.run_epoch(session, valid_model, data["valid"], batch_size = batch_size, separate=True)
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

                # early stop if generalization loss is worst than args.early_stopping
                if args.early_stopping > 0:
                    if ((valid_loss / early_stop_best_loss) - 1.0) > args.early_stopping:
                        print 'Early stopping criteria reached: {}'.format(args.early_stopping)
                        break

            if not saved_flag:
                saver.save(session, os.path.join(args.model_dir, model_name + model_suffix))

            # set loss axis max to 20
            axes = plt.gca()
            axes.set_ylim([0, 2])
            plt.plot([t[0] for t in train_losses], [t[1] for t in train_losses])
            plt.plot([t[0] for t in valid_losses], [t[1] for t in valid_losses])
            plt.legend(['Train Loss', 'Validation Loss'])
            plt.savefig(os.path.join(args.charts_dir, model_name + charts_suffix))
            plt.clf()
            # print "Saved graph"

            print "Model {}, Loss: {}".format(model_name, early_stop_best_loss)

    if args.test:

        with tf.Graph().as_default(), tf.Session() as session:

            with tf.variable_scope(model_name, reuse=True if args.train else None):
                test_model = model_class(default_config)

            saver = tf.train.Saver(tf.all_variables())
            model_path = os.path.join(args.model_dir, model_name + model_suffix)
            saver.restore(session, model_path)

            # Deterministic Testing
            if args.test: 
                test_loss, test_probs = util.run_epoch(session, test_model, 
                                                       data["test"], 
                                                       training=False, testing=True, separate=True)

                print 'Testing Loss ({}): {}'.format(model_name, test_loss)
                nottingham_util.seperate_accuracy(test_probs, pickle['test'], default_config)
