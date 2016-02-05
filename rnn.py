import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    

import midi_util
import sampling
import preprocess
from model import Model

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

if __name__ == '__main__':
    np.random.seed(1)      

    parser = argparse.ArgumentParser(description='Music RNN')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('-n', '--model_name', type=str, default='default_model')
    parser.add_argument('-d', '--model_dir', type=str, default='models')
    parser.add_argument('-s', '--model_suffix', type=str, default='_jsb_chorales.model')
    parser.add_argument('-t', '--temp', type=float, default=0.5)
    parser.add_argument('--data_dir', type=str, default='data/JSBChorales')
    args = parser.parse_args()

    data = preprocess.load_data(args.data_dir)
    print 'Finished loading data, input dim: {}'.format(data["input_dim"])

    num_epochs = 800
    default_config = {
        "input_dim": data["input_dim"],
        "hidden_size": 150,
        "num_layers": 1,
    } 

    def set_config(config, name):
        return dict(config, **{
            "batch_size": data[name]["data"].shape[1],
            "seq_length": data[name]["data"].shape[0]
        })

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        if args.train:
            best_config = None
            best_valid_loss = None
            best_model_name = None

            for num_layers in [1]:
                for hidden_size in [150]:
                    for learning_rate in [1e-2]:

                        model_name = "nl_" + str(num_layers) + \
                                     "_hs_" + str(hidden_size) + \
                                     "_lr_" + str(learning_rate).replace(".", "p")
                        config = {
                            "input_dim": data["input_dim"],
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                        }

                        with tf.variable_scope(model_name, reuse=None):
                            train_model = Model(set_config(config, "train"))
                        with tf.variable_scope(model_name, reuse=True):
                            valid_model = Model(set_config(config, "valid"))

                        saver = tf.train.Saver(tf.all_variables())
                        tf.initialize_all_variables().run()

                        # training
                        learning_rate = learning_rate
                        train_model.assign_lr(session, learning_rate)
                        for i in range(num_epochs):
                            loss = run_epoch(session, train_model, 
                                             data["train"]["data"], data["train"]["targets"], 
                                             training=True)
                            if i % 10 == 0:
                                valid_loss = run_epoch(session, valid_model, 
                                    data["valid"]["data"], data["valid"]["targets"])
                                print 'Epoch: {}, Train Loss: {}, Valid Loss: {}'.format(i, loss, valid_loss)

                        saver.save(session, os.path.join(args.model_dir, model_name + args.model_suffix))
                        print "Saved model"


                        valid_loss = run_epoch(session, valid_model, 
                            data["valid"]["data"], data["valid"]["targets"])
                        print "Model {} Loss: {}".format(model_name, valid_loss)
                        if best_valid_loss == None or valid_loss < best_valid_loss:
                            print "Found best new model: {}".format(model_name)
                            best_valid_loss = valid_loss
                            best_config = config
                            best_model_name = model_name

            print 'Best config ({}): {}'.format(best_model_name, best_config)

            with tf.variable_scope(best_model_name, reuse=True):
                test_model = Model(set_config(best_config, "test"))

            # testing
            test_loss = run_epoch(session, test_model, data["test"]["data"], data["test"]["targets"])
            print 'Testing Loss ({}): {}'.format(best_model_name, test_loss)

            with tf.variable_scope(best_model_name, reuse=True):
                sampling_model = Model(dict(best_config, **{
                    "batch_size": 1,
                    "seq_length": 1
                }))

        else:
            with tf.variable_scope(args.model_name, reuse=None):
                sampling_model = Model(dict(default_config, **{
                    "batch_size": 1,
                    "seq_length": 1
                }))
            saver = tf.train.Saver(tf.all_variables())
            model_path = os.path.join(args.model_dir, args.model_name + args.model_suffix)
            saver.restore(session, model_path)

        # start with the first chord
        chord = data["train"]["data"][0, 0, :]
        seq = [chord]
        state = sampling_model.initial_state.eval()

        max_seq_len = 200
        for i in range(max_seq_len):
            seq_input = np.reshape(chord, [1, 1, data["input_dim"]])
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [data["input_dim"]])
            # chord = sampling.sample_notes_static(probs, num_notes=4)
            chord = sampling.sample_notes_dynamic(probs, min_prob=args.temp)
            # if the "end-of-sequence token reached, exit"
            # if chord[-1] > 0:
            #     print "Sequence length: {}".format(i)
            #     continue
            sampling.visualize_probs(probs)
            seq.append(chord)
         
        print seq
        midi_util.dump_sequence_to_midi(seq, "best.midi")


