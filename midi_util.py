import sys, os
import numpy as np
from fractions import gcd

import midi

RANGE = 128

def parse_midi_to_sequence(input_filename, time_step, verbose=False):
    sequence = []
    pattern = midi.read_midifile(input_filename)

    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    if verbose:
        print "Track resolution: {}".format(pattern.resolution)
        print "Number of tracks: {}".format(len(pattern))

    def round_tick(tick):
        return int(round(tick/float(time_step)) * time_step)

    if verbose:
        print "Time step: {}".format(time_step)

    # Track ingestion stage
    midi_errors = 0
    notes = { n: [] for n in range(RANGE) }
    track_ticks = 0
    for track in pattern:
        current_tick = 0
        for msg in track:
            # ignore all end of track events
            if isinstance(msg, midi.EndOfTrackEvent):
                continue

            if msg.tick > 0: 
                current_tick += msg.tick

            # velocity of 0 is equivalent to note off, so treat as such
            if isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() != 0:
                if len(notes[msg.get_pitch()]) > 0 and \
                   len(notes[msg.get_pitch()][-1]) != 2:
                    if verbose:
                        print "Warning: double NoteOn encountered, deleting the first"
                        print msg
                        midi_errors += 1
                else:
                    notes[msg.get_pitch()] += [[current_tick]]
            elif isinstance(msg, midi.NoteOffEvent) or \
                (isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() == 0):
                # sanity check: no notes end without being started
                if len(notes[msg.get_pitch()][-1]) != 1:
                    if verbose:
                        print "Warning: skipping NoteOff Event with no corresponding NoteOn"
                        print msg
                        midi_errors += 1
                else: 
                    notes[msg.get_pitch()][-1] += [current_tick]

        track_ticks = max(current_tick, track_ticks)

    if verbose:
        print "Detected {} midi errors".format(midi_errors)

    track_ticks = round_tick(track_ticks)
    if verbose:
        print "Track ticks (rounded): {} ({} time steps)".format(track_ticks, track_ticks/time_step)

    sequence = np.zeros((track_ticks/time_step, RANGE))
    
    # Rounding stage
    skipped_count = 0
    for note in notes:
        for (start, end) in notes[note]:
            if end - start <= time_step/2:
                skipped_count += 1
            else:
                start_t = round_tick(start) / time_step
                end_t = round_tick(end) / time_step
                if start_t == end_t:
                    skipped_count += 1
                else:
                    sequence[start_t:end_t, note] = 1

    return sequence

def dump_sequence_to_midi(sequence, output_filename, time_step, 
                          resolution, verbose=False):
    if verbose:
        print "Dumping sequence to MIDI file: {}".format(output_filename)
        print "Resolution: {}".format(resolution)
        print "Time Step: {}".format(time_step)

    pattern = midi.Pattern(resolution=resolution)
    track = midi.Track()

    # reshape to (SEQ_LENGTH X NUM_DIMS)
    sequence = np.reshape(sequence, [-1, RANGE])

    time_steps = sequence.shape[0]
    if verbose:
        print "Total number of time steps: {}".format(time_steps)

    steps_passed = 1
    notes_on = { n: False for n in range(RANGE) }
    for seq_idx in range(time_steps):
        notes = np.nonzero(sequence[seq_idx, :])[0].tolist()

        # this tick will only be assigned to first NoteOn/NoteOff in
        # this time_step
        tick = steps_passed * time_step

        # NoteOffEvents come first so they'll have the tick value
        # go through all notes that are currently on and see if any
        # turned off
        for n in notes_on:
            if notes_on[n] and n not in notes:
                track.append(midi.NoteOffEvent(tick=tick, pitch=n))
                tick, steps_passed = 0, 0
                notes_on[n] = False

        # Turn on any notes that weren't previously on
        for note in notes:
            if not notes_on[note]:
                track.append(midi.NoteOnEvent(tick=tick, pitch=note, velocity=70))
                tick, steps_passed = 0, 0
                notes_on[note] = True

        steps_passed += 1

    # flush out notes
    tick = steps_passed * time_step
    for n in notes_on:
        track.append(midi.NoteOffEvent(tick=tick, pitch=n))
        tick = 0
        notes_on[n] = False

    # track.append(midi.EndOfTrackEvent())
    pattern.append(track)
    midi.write_midifile(output_filename, pattern)

def chord_on(notes=[]):
    chord = np.zeros(RANGE, dtype=np.float32)
    for n in notes:
        chord[n] = 1.0
    return chord 

def cmaj():
    return chord_on((72, 76, 79))

def amin():
    return chord_on((72, 76, 81))

def fmaj():
    return chord_on((72, 77, 81))

def gmaj():
    return chord_on((74, 79, 83))

def i_vi_iv_v(n):
    return  [[cmaj(), amin(), fmaj(), gmaj()] * n]

if __name__ == '__main__':
    # parse_midi_directory("data/JSBChorales/train", 120)
    # parse_midi_directory("data/Nottingham/train", 120)
    pass
