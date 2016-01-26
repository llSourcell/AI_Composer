import os
 
import numpy as np
import tensorflow as tf    
from tensorflow.models.rnn import rnn    
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell    

from midi_util import midi_to_sequence, sequence_to_midi
from model import Model

DATA_DIR = 'data/JSBChorales'

def prepare_data(dataset_name):
  train_path = os.path.join(DATA_DIR, dataset_name)
  train_files = [ os.path.join(train_path, f) for f in os.listdir(train_path)
                  if os.path.isfile(os.path.join(train_path, f)) ] 
  train_seq = [ midi_to_sequence(f) for f in train_files ]

  # number of examples 
  N = len(train_seq)
  # sequence length is the longest sequence
  seq_length = max(len(t) for t in train_seq)
  # dimension of input
  max_note = max(max([max(c) for c in seq if len(c) > 0]) for seq in train_seq)
  min_note = min(min([min(c) for c in seq if len(c) > 0]) for seq in train_seq)

  return (train_seq, seq_length, max_note)

if __name__ == '__main__':
  np.random.seed(1)      

  train_seqs, train_len, train_max = prepare_data('train')
  valid_seqs, valid_len, valid_max = prepare_data('valid')
  test_seqs, test_len, test_max = prepare_data('valid')

  D = max(train_max, valid_max, test_max)+1

  print 'Dimension: {}'.format(D)

  # TRAINING
  # (sequence length) X (num sequences) X (dimension)
  train_data = np.zeros([train_len, len(train_seqs), D], dtype=np.float32)
  for idx, seq in enumerate(train_seqs):
      for c_idx, chord in enumerate(seq):
          train_data[c_idx, idx, chord] = 1

  # roll back the time steps axis to get the target of each example
  train_targets = np.roll(train_data, -1, axis=0)
  # set final time step targets to 0
  train_targets[-1, :, :] = 0 

  model = Model(len(train_seqs), D, train_len, hidden_size=100, initialization=0.1)
  model.train(train_data, train_targets)

  # TESTING
  test_data = np.zeros([test_len, len(test_seqs), D], dtype=np.float32)
  for idx, seq in enumerate(test_seqs):
      for c_idx, chord in enumerate(seq):
          test_data[c_idx, idx, chord] = 1

  # roll back the time steps axis to get the target of each example
  test_targets = np.roll(test_data, -1, axis=0)
  # set final time step targets to 0
  test_targets[-1, :, :] = 0 

  model = Model(len(test_seqs), D, test_len, hidden_size=100, initialization=0.1)
  print model.test(test_data, test_targets)

  # sampling
  # model = Model(N, D, seq_length, hidden_size=100, 
  #               initialization=0.1, sampling_mode=True)
  # seq = model.sample(np.zeros(D), temperature=0.2)
  # chords = [np.nonzero(c)[0].tolist() for c in seq]
  # sequence_to_midi(chords, "gen1.midi")
