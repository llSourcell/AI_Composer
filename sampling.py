import numpy as np
from pprint import pprint

import midi_util


class Sampler(object):

    def __init__(self, min_prob=0.5, num_notes = 4, method = 'sample', verbose=False):
        self.min_prob = min_prob
        self.num_notes = num_notes
        self.method = method
        self.verbose = verbose

    def visualize_probs(self, probs):
        if not self.verbose:
            return
        print 'Highest four probs: '
        pprint(sorted(list(enumerate(probs)), key=lambda x: x[1], 
               reverse=True)[:4])

    def sample_notes_prob(self, probs, max_notes=-1):
        """ Samples all notes that are over a certain probability"""
        self.visualize_probs(probs)
        top_idxs = list()
        for idx in probs.argsort()[::-1]:
            if max_notes > 0 and len(top_idxs) >= max_notes:
                break
            if probs[idx] < self.min_prob:
                break
            top_idxs.append(idx)
        chord = np.zeros([len(probs)], dtype=np.int32)
        chord[top_idxs] = 1.0
        return chord

    def sample_notes_static(self, probs):
        top_idxs = probs.argsort()[-self.num_notes:][::-1]
        chord = np.zeros([len(probs)], dtype=np.int32)
        chord[top_idxs] = 1.0
        return chord

    def sample_notes_bernoulli(self, probs):
        chord = np.zeros([len(probs)], dtype=np.int32)
        for note, prob in enumerate(probs):
            if np.random.binomial(1, prob) > 0:
                chord[note] = 1
        return chord

    def sample_notes(self, probs):
        """ Samples a static amount of notes from probabilities by highest prob """
        self.visualize_probs(probs)
        if self.method == 'sample':
            return self.sample_notes_bernoulli(probs)
        elif self.method == 'static':
            return self.sample_notes_static(probs)
        elif self.method == 'min_prob':
            return self.sample_notes_prob(probs)
        else:
            raise Exception("Unrecognized method: {}".format(self.method))
