import sys, os
import numpy as np
from fractions import gcd

import midi

RANGE = 128

class NonDivisibleTimeStepException(Exception):
    pass

def parse_midi_to_sequence(input_filename, round_to=-1, min_time_step=-1, verbose=False):
    sequence = []
    pattern = midi.read_midifile(input_filename)

    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    if verbose:
        print "Track resolution: {}".format(pattern.resolution)
        print "Number of tracks: {}".format(len(pattern))

    def round_tick(tick):
        if round_to < 0:
            return tick
        else:
            return int(round(tick/float(round_to)) * round_to)

    if min_time_step < 0:
        # find min_time_step automatically by finding gcd
        min_time_step = pattern.resolution
        for track in pattern:
            for msg in track:
                # we don't care about the tick values for non notes
                if not isinstance(msg, midi.NoteOnEvent) and \
                   not isinstance(msg, midi.NoteOffEvent):
                    continue 

                tick = round_tick(msg.tick)
                if tick > 0:
                    min_time_step = gcd(min_time_step, tick)

    if verbose:
        print "Minimum time step: {}".format(min_time_step)

    def union_tracks(track1, track2):
        if track1.shape[1] != track2.shape[1]:
            raise Exception("Track dimensions \
                don't match: {} vs {}".track1.shape, track2.shape)
        if track1.shape[0] > track2.shape[0]:
            unioned = track1
            track = track2
        else:
            unioned = track2
            track = track1

        for i in range(track.shape[0]): 
            unioned[i, :] = np.minimum(unioned[i, :] + track[i, :], 1)

        return unioned
    
    parsed = None
    for track in pattern:
        seq = list()
        current_chord = np.zeros(RANGE*2)
        for msg in track:
            if isinstance(msg, midi.EndOfTrackEvent):
                continue 

            tick = round_tick(msg.tick)
            if tick > 0:
                if tick % min_time_step != 0:
                    raise NonDivisibleTimeStepException("Non-divisible time step \
                        encountered in MIDI: {}".format(tick))
                num_time_steps = tick / min_time_step
                seq.append(current_chord)
                for i in range(num_time_steps-1):
                    seq.append(np.zeros(RANGE*2))
                current_chord = np.zeros(RANGE*2)

            if isinstance(msg, midi.NoteOnEvent):
                # velocity of 0 is equivalent to note off, so treat as such
                if msg.get_velocity() == 0:
                    current_chord[msg.get_pitch() + RANGE] = 1
                else:
                    current_chord[msg.get_pitch()] = 1
            elif isinstance(msg, midi.NoteOffEvent):
                current_chord[msg.get_pitch() + RANGE] = 1
            else:
                pass

        # flush out remaining chord
        seq.append(current_chord)
        if parsed == None:
            parsed = np.vstack(seq)
        else:
            parsed = union_tracks(parsed, np.vstack(seq))

    if verbose:
        print "Total time steps: {}".format(parsed.shape[0])

    return min_time_step, parsed

def parse_midi_directory(input_dir, round_to=-1, max_seq_len=-1, min_time_step=-1, 
                         verbose=False):
    # TODO: only uses the first max_seq_len... change this line?
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    raw_sequences = list()
    num_skipped = 0
    for f in files:
        try: 
            _, seq = parse_midi_to_sequence(f, round_to=round_to, min_time_step=120)
        except NonDivisibleTimeStepException:
            if verbose:
                print "Skipped track: {}".format(f)
            num_skipped += 1
            continue
        raw_sequences.append(seq)

    if verbose:
        print "Number of Sequences: {} ({} skipped due to time step)".format(len(raw_sequences), num_skipped)
    dims = raw_sequences[0].shape[1]

    if max_seq_len < 0:
        # make all sequences length of max sequence
        max_seq_len = max(s.shape[0] for s in raw_sequences)

    min_seq_len = max_seq_len/2

    if verbose:
        avg_seq_len = sum(s.shape[0] for s in raw_sequences) / len(raw_sequences)
        print "Average Sequence Length: {}".format(avg_seq_len)
        print "Max Sequence Length: {}".format(max_seq_len)

    sequences = list()
    for i in range(len(raw_sequences)):
        # ignore any sequences that are too short
        if raw_sequences[i].shape[0] < min_seq_len:
            continue
        if raw_sequences[i].shape[0] <= max_seq_len:
            seq = raw_sequences[i].copy()
            seq.resize((max_seq_len, dims))
            sequences.append(seq)
        elif raw_sequences[i].shape[0] > max_seq_len:
            # split up the sequences into max_seq_len each, except for when
            # the split is less than min_seq_len
            for j in range(raw_sequences[i].shape[0] / max_seq_len + 1):
                seq = raw_sequences[i][j*max_seq_len:(j+1)*max_seq_len].copy()
                break # TODO: change this line?
                # if seq.shape[0] < min_seq_len:
                #     continue
                # seq.resize((max_seq_len, dims))
                # sequences.append(seq)

    batch = np.dstack(sequences)
    # swap axes so that shape is (SEQ_LENGTH X BATCH_SIZE X INPUT_DIM)
    batch = np.swapaxes(batch, 1, 2)

    return batch

def dump_sequence_to_midi(sequence, output_filename, min_time_step=20, 
                          resolution=100):
    pattern = midi.Pattern(resolution=resolution)
    track = midi.Track()

    # set the bpm
    # set_tempo = midi.SetTempoEvent()
    # set_tempo.set_bpm(bpm)
    # track.append(set_tempo)

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
                tick = steps_skipped * min_time_step 
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
    # a = parse_midi_to_sequence("nottingham_sample.midi", verbose=True, round_to=120)
    # dump_sequence_to_midi(a, "sample.midi", min_time_step=120, resolution=480) 
    parse_midi_directory("data/Nottingham/test", round_to=40, min_time_step=120, 
                         max_seq_len = 200, verbose=True)
    # parse_midi_directory("data/JSBChorales/train", round_to=10, verbose=True)

