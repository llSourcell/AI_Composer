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

        # if there are notes
        for i in range(len(idxs)):
            if i == 0:
                tick = steps_skipped * time_step
            else: 
                tick = 0

            idx = idxs[i]
            if idx < RANGE:
                track.append(midi.NoteOnEvent(tick=tick, pitch=idx, velocity=90))
            else: 
                track.append(midi.NoteOffEvent(tick=tick, pitch=idx-RANGE))

        steps_skipped = 1

    track.append(midi.EndOfTrackEvent())
    pattern.append(track)
    midi.write_midifile(output_filename, pattern)
        
if __name__ == '__main__':
    a = parse_midi_to_sequence("data/JSBChorales/train/10.mid")
    print np.nonzero(a[-1])[0]
    dump_sequence_to_midi(a, "train_10.midi", time_step=120)

############################
# VERSION 1 function below #
# TODO(yoavz): depcrecate  # 
############################

def midi_to_sequence(input_filename):
    """ Reads a midi file into a sequence """
    """ Warning: algorithm is hacky and not verified """

    sequence = []
    pattern = midi.read_midifile(input_filename)

    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    chord = set()
    for msg in pattern[0]:
        # The first nonzero tick indicates a new note
        # TODO(yoavz): generalize this to all midi instead of just JSBChorales
        if msg.tick > 0:
            measures = msg.tick / 120
            for i in range(measures): 
                sequence.append(sorted(list(chord)))

        if isinstance(msg, midi.NoteOnEvent):
            chord.add(msg.data[0])
        elif isinstance(msg, midi.NoteOffEvent):
            # sanity check the note is in the chord
            assert msg.data[0] in chord
            chord.remove(msg.data[0])
            
    return sequence

def sequence_to_midi(sequence, output_filename):
    """ Dumps the pattern from a sequence to an output file """ 
    """ Warning: algorithm is hacky and not verified """

    pattern = midi.Pattern(resolution=100)
    track = midi.Track()
    pattern.append(track)

    history = set()
    time_passed = 0
    for chord in sequence:
        first_tick = False

        for note in history:
            if note not in chord:
                if not first_tick: 
                    track.append(midi.NoteOffEvent(tick=time_passed, pitch=note))
                    first_tick = True
                else:
                    track.append(midi.NoteOffEvent(tick=0, pitch=note))

        for note in chord:
            if note not in history:
                if not first_tick:
                    track.append(midi.NoteOnEvent(tick=time_passed, 
                                                  velocity=90, 
                                                  pitch=note))
                    first_tick = True
                else:
                    track.append(midi.NoteOnEvent(tick=0, 
                                                  velocity=90, 
                                                  pitch=note))

        if not first_tick:
            time_passed += 120
        else:
            time_passed = 120

        # reset the history
        history = set(chord)
    
    # flush out the last chords in the sequence
    for idx, note in enumerate(chord):
        if idx == 0:
            track.append(midi.NoteOffEvent(tick=time_passed, pitch=note))
        else:
            track.append(midi.NoteOffEvent(tick=0, pitch=note))

    midi.write_midifile(output_filename, pattern)

def i_vi_iv_v(n):

    # cmaj = (60, 72, 76, 79)
    # amin = (57, 72, 76, 81)
    # fmaj = (53, 72, 77, 81)
    # gmaj = (55, 74, 79, 83)

    cmaj = (72, 76, 79)
    amin = (72, 76, 81)
    fmaj = (72, 77, 81)
    gmaj = (74, 79, 83)

    progression = [cmaj, amin, fmaj, gmaj]
    return progression * n
