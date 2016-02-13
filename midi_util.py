import sys, os
import numpy as np

import midi

RANGE = 128

def parse_midi_to_sequence(input_filename): 
    """ NOTE: Only works on ONE track. """
    sequence = []
    pattern = midi.read_midifile(input_filename)

    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    # get the minimum time step
    min_time_step = sys.maxint
    for msg in pattern[0]:
        if msg.tick > 0:
            min_time_step = min(min_time_step, msg.tick)

    seq = list()
    current_chord = np.zeros(RANGE*2)
    for msg in pattern[0]:
        if msg.tick > 0:
            if msg.tick % min_time_step != 0:
                raise Exception("Non-divisible time step encountered in MIDI")
            num_time_steps = msg.tick / min_time_step
            seq.append(current_chord)
            for i in range(num_time_steps-1):
                seq.append(np.zeros(RANGE*2))
            current_chord = np.zeros(RANGE*2)

        if isinstance(msg, midi.NoteOnEvent):
            current_chord[msg.data[0]] = 1
        elif isinstance(msg, midi.NoteOffEvent):
            current_chord[msg.data[0]+RANGE] = 1
        elif isinstance(msg, midi.ProgramChangeEvent):
            continue
        elif isinstance(msg, midi.EndOfTrackEvent):
            continue
        else:
            raise Exception("Unknown MIDI Event encountered: {}".format(msg))

    # flush out remaining chord
    seq.append(current_chord)

    return np.vstack(seq)

def parse_midi_directory(input_dir):
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    sequences = [parse_midi_to_sequence(f) for f in files]
    dims = sequences[0].shape[1]

    # make all sequences length of max sequence
    max_seq_len = max(s.shape[0] for s in sequences)
    for i in range(len(sequences)):
        sequences[i].resize((max_seq_len, dims))

    batch = np.dstack(sequences)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
    batch = np.swapaxes(batch, 1, 2)

    return batch

def dump_sequence_to_midi(sequence, output_filename, time_step=120, max_note_len=4):
    pattern = midi.Pattern(resolution=100)
    track = midi.Track()

    # reshape to (SEQ_LENGTH X NUM_DIMS)
    sequence = np.reshape(sequence, [-1, RANGE*2])

    steps_skipped = 1
    for seq_idx in range(sequence.shape[0]):
        idxs = np.nonzero(sequence[seq_idx, :])[0].tolist()
        # if there aren't any notes, skip this time step
        if len(idxs) == 0:
            steps_skipped += 1
            continue

        # NoteOffEvents come first so they'll have the tick value
        idxs = sorted(idxs, reverse=True)

        # if there are notes
        for i in range(len(idxs)):
            if i == 0:
                tick = steps_skipped * time_step
            else: 
                tick = 0

            idx = idxs[i]
            if idx >= RANGE:
                track.append(midi.NoteOffEvent(tick=tick, pitch=idx-RANGE))
            else: 
                track.append(midi.NoteOnEvent(tick=tick, pitch=idx, velocity=90))

        steps_skipped = 1

    track.append(midi.EndOfTrackEvent())
    pattern.append(track)
    midi.write_midifile(output_filename, pattern)

def chord_on(notes=[]):
    chord = np.zeros(RANGE*2, dtype=np.float32)
    for n in notes:
        chord[n] = 1.0
    return chord 

def chord_off(chord):
    return np.roll(chord, RANGE)

def cmaj():
    return chord_on((72, 76, 79))

def amin():
    return chord_on((72, 76, 81))

def fmaj():
    return chord_on((72, 77, 81))

def gmaj():
    return chord_on((74, 79, 83))

def i_vi_iv_v(n):
    i = cmaj()
    vi = chord_off(cmaj()) + amin()
    iv = chord_off(amin()) + fmaj()
    v = chord_off(fmaj()) + gmaj()
    i_transition = chord_off(gmaj()) + cmaj()

    return [i, vi, iv, v] + \
           [i_transition, vi, iv, v] * (n-1) + \
           [i, chord_on(), chord_on(), chord_off(i)]
        
if __name__ == '__main__':
    a = parse_midi_to_sequence("data/JSBChorales/train/10.mid")
    print np.nonzero(a[-1])[0]
    dump_sequence_to_midi(a, "train_10.midi", time_step=120)
