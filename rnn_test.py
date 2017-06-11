import os, sys
import argparse
import cPickle

import numpy as np
import tensorflow as tf    

import util
import nottingham_util
from model import Model, NottinghamModel, NottinghamSeparate
from rnn import DefaultConfig

if __name__ == '__main__':
    np.random.seed()      

    parser = argparse.ArgumentParser(description='Script to test a models performance against the test set')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--seperate', action='store_true', default=False)
    parser.add_argument('--choice', type=str, default='melody',
                        choices = ['melody', 'harmony'])
    args = parser.parse_args()

    with open(args.config_file, 'r') as f: 
        config = cPickle.load(f)

    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        with open(nottingham_util.PICKLE_LOC, 'r') as f:
            pickle = cPickle.load(f)
        if args.seperate:
            model_class = NottinghamSeparate
            test_data = util.batch_data(pickle['test'], time_batch_len = 1, 
                max_time_batches = -1, softmax = True)
            r = nottingham_util.NOTTINGHAM_MELODY_RANGE
            if args.choice == 'melody':
                print "Using only melody"
                new_data = []
                for batch_data, batch_targets in test_data:
                    new_data.append(([tb[:, :, :r] for tb in batch_data],
                                     [tb[:, :, 0] for tb in batch_targets]))
                test_data = new_data
            else:
                print "Using only harmony"
                new_data = []
                for batch_data, batch_targets in test_data:
                    new_data.append(([tb[:, :, r:] for tb in batch_data],
                                     [tb[:, :, 1] for tb in batch_targets]))
                test_data = new_data
        else:
            model_class = NottinghamModel
            # use time batch len of 1 so that every target is covered
            test_data = util.batch_data(pickle['test'], time_batch_len = 1, 
                max_time_batches = -1, softmax = True)
    else:
        raise Exception("Other datasets not yet implemented")
        
    print config

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            test_model = model_class(config, training=False)

        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        saver.restore(session, model_path)
        
        test_loss, test_probs = util.run_epoch(session, test_model, test_data, 
            training=False, testing=True)
        print 'Testing Loss: {}'.format(test_loss)

        if config.dataset == 'softmax':
            if args.seperate:
                nottingham_util.seperate_accuracy(test_probs, test_data, num_samples=args.num_samples)
            else:
                nottingham_util.accuracy(test_probs, test_data, num_samples=args.num_samples)

        else:
            util.accuracy(test_probs, test_data, num_samples=50)

    sys.exit(1)
