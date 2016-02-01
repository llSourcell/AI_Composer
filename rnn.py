import os, sys
import argparse
 
import numpy as np
import tensorflow as tf    
from pprint import pprint

from midi_util import midi_to_sequence, sequence_to_midi, i_vi_iv_v
from preprocess import load_data
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

    data = load_data()
    print 'Finished loading data'

    num_epochs = 1000
    default_config = {
        "input_dim": data["input_dim"],
        "hidden_size": 100,
        "num_layers": 2,
    } 

    def set_config(config, name):
        return dict(config, **{
            "batch_size": data[name]["data"].shape[1],
            "seq_length": data[name]["data"].shape[0]
        })

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        best_config = None
        best_valid_loss = None
        best_model_name = None

        for num_layers in [1, 2]:
            for hidden_size in [100, 200]:
                for learning_rate in [0.1, 0.01]:

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
                    
                    tf.initialize_all_variables().run()

                    # training
                    learning_rate = learning_rate
                    train_model.assign_lr(session, learning_rate)
                    for i in range(num_epochs):
                        loss = run_epoch(session, train_model, 
                                         data["train"]["data"], data["train"]["targets"], 
                                         training=True)
                        if i % 10 == 0:
                            print 'Epoch: {}, Loss: {}'.format(i, loss)

                    saver = tf.train.Saver(tf.all_variables())
                    saver.save(session, os.path.join("models", model_name + "_jsb_chorales.model"))
                    print "Saved model"

                    with tf.variable_scope(model_name, reuse=True):
                        valid_model = Model(set_config(config, "valid"))

                    loss = run_epoch(session, valid_model, 
                                     data["valid"]["data"], data["valid"]["targets"])
                    print "Model {} Loss: {}".format(model_name, loss)
                    if best_valid_loss == None or loss < best_valid_loss:
                        print "Found best new model: {}".format(model_name)
                        best_valid_loss = loss
                        best_config = config
                        best_model_name = model_name

        with tf.variable_scope(best_model_name, reuse=True):
            test_model = Model(set_config(best_config, "test"))
        
        # testing
        loss = run_epoch(session, test_model, data["test"]["data"], data["test"]["targets"])
        print 'Testing Loss (Model {}): {}'.format(model_name, loss)

        # sampling
        def sample_notes_from_probs(probs, n):
            reshaped = np.reshape(probs, [data["input_dim"]])
            top_idxs = reshaped.argsort()[-n:][::-1]
            chord = np.zeros([data["input_dim"]], dtype=np.int32)
            chord[top_idxs] = 1.0
            return chord

        def visualize_probs(probs, targets):
            probs = list(probs[0])
            targets = list(targets)
            # print 'First four notes: '
            # pprint(zip(probs, targets))
            # print 'Highest four probs: '
            # pprint(sorted(list(enumerate(probs)), key=lambda x: x[1], 
            #                                       reverse=True)[:4])

        with tf.variable_scope(best_model_name, reuse=True):
            sampling_model = Model(dict(best_config, **{
                "batch_size": 1,
                "seq_length": 1
            }))

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
            chord = sample_notes_from_probs(probs, 4)
            # if the "end-of-sequence token reached, exit"
            if chord[-1] > 0:
                print "Sequence length: {}".format(i)
                continue
            # visualize_probs(probs, data["train"]["targets"][i, 0, :])
            seq.append(chord)
         
        chords = [np.nonzero(c)[0].tolist() for c in seq]
        sequence_to_midi(chords, "best.midi")


