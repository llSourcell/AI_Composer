import numpy as np
from pprint import pprint

def sample_notes_static(probs, num_notes=4):
    """ Samples a static amount of notes from probabilities by highest prob """
    top_idxs = probs.argsort()[-num_notes:][::-1]
    chord = np.zeros([len(probs)], dtype=np.int32)
    chord[top_idxs] = 1.0
    return chord

def sample_notes_dynamic(probs, min_prob=0.5, max_notes=4):
    """ Samples all notes that are over a certain probability"""
    top_idxs = list()
    for idx in probs.argsort()[::-1]:
        if len(top_idxs) >= max_notes:
            break
        if probs[idx] < min_prob:
            break
        top_idxs.append(idx)
    chord = np.zeros([len(probs)], dtype=np.int32)
    chord[top_idxs] = 1.0
    return chord


def visualize_probs(probs):
    # print 'First four notes: '
    # pprint(zip(probs, targets))
    # print 'Highest four probs: '
    # pprint(sorted(list(enumerate(probs)), key=lambda x: x[1], 
    #               reverse=True)[-4:])
    pass
