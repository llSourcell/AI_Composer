import sys, os
from collections import defaultdict
import numpy as np
import midi

RANGE = 128

def round_tick(tick, time_step):
    return int(round(tick/float(time_step)) * time_step)

def ingest_notes(track, verbose=False):

    notes = { n: [] for n in range(RANGE) }
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
            else:
                notes[msg.get_pitch()] += [[current_tick]]
        elif isinstance(msg, midi.NoteOffEvent) or \
            (isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() == 0):
            # sanity check: no notes end without being started
            if len(notes[msg.get_pitch()][-1]) != 1:
                if verbose:
                    print "Warning: skipping NoteOff Event with no corresponding NoteOn"
                    print msg
            else: 
                notes[msg.get_pitch()][-1] += [current_tick]

    return notes, current_tick

def round_notes(notes, track_ticks, time_step, R=None, O=None):
    if not R:
        R = RANGE
    if not O:
        O = 0

    sequence = np.zeros((track_ticks/time_step, R))
    disputed = { t: defaultdict(int) for t in range(track_ticks/time_step) }
    for note in notes:
        for (start, end) in notes[note]:
            start_t = round_tick(start, time_step) / time_step
            end_t = round_tick(end, time_step) / time_step
            # normal case where note is long enough
            if end - start > time_step/2 and start_t != end_t:
                sequence[start_t:end_t, note - O] = 1
            # cases where note is within bounds of time step 
            elif start > start_t * time_step:
                disputed[start_t][note] += (end - start)
            elif end <= end_t * time_step:
                disputed[end_t-1][note] += (end - start)
            # case where a note is on the border 
            else:
                before_border = start_t * time_step - start
                if before_border > 0:
                    disputed[start_t-1][note] += before_border
                after_border = end - start_t * time_step
                if after_border > 0 and end < track_ticks:
                    disputed[start_t][note] += after_border

    # solve disputed
    for seq_idx in range(sequence.shape[0]):
        if np.count_nonzero(sequence[seq_idx, :]) == 0 and len(disputed[seq_idx]) > 0:
            # print seq_idx, disputed[seq_idx]
            sorted_notes = sorted(disputed[seq_idx].items(),
                                  key=lambda x: x[1])
            max_val = max(x[1] for x in sorted_notes)
            top_notes = filter(lambda x: x[1] >= max_val, sorted_notes)
            for note, _ in top_notes:
                sequence[seq_idx, note - O] = 1

    return sequence

def parse_midi_to_sequence(input_filename, time_step, verbose=False):
    sequence = []
    pattern = midi.read_midifile(input_filename)

    if len(pattern) < 1:
        raise Exception("No pattern found in midi file")

    if verbose:
        print "Track resolution: {}".format(pattern.resolution)
        print "Number of tracks: {}".format(len(pattern))
        print "Time step: {}".format(time_step)

    # Track ingestion stage
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
                else:
                    notes[msg.get_pitch()] += [[current_tick]]
            elif isinstance(msg, midi.NoteOffEvent) or \
                (isinstance(msg, midi.NoteOnEvent) and msg.get_velocity() == 0):
                # sanity check: no notes end without being started
                if len(notes[msg.get_pitch()][-1]) != 1:
                    if verbose:
                        print "Warning: skipping NoteOff Event with no corresponding NoteOn"
                        print msg
                else: 
                    notes[msg.get_pitch()][-1] += [current_tick]

        track_ticks = max(current_tick, track_ticks)

    track_ticks = round_tick(track_ticks, time_step)
    if verbose:
        print "Track ticks (rounded): {} ({} time steps)".format(track_ticks, track_ticks/time_step)

    sequence = round_notes(notes, track_ticks, time_step)

    return sequence

class MidiWriter(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.note_range = RANGE

    def note_off(self, val, tick):
        self.track.append(midi.NoteOffEvent(tick=tick, pitch=val))
        return 0

    def note_on(self, val, tick):
        self.track.append(midi.NoteOnEvent(tick=tick, pitch=val, velocity=70))
        return 0

    def dump_sequence_to_midi(self, sequence, output_filename, time_step, 
                              resolution, metronome=24):
        if self.verbose:
            print "Dumping sequence to MIDI file: {}".format(output_filename)
            print "Resolution: {}".format(resolution)
            print "Time Step: {}".format(time_step)

        pattern = midi.Pattern(resolution=resolution)
        self.track = midi.Track()

        # metadata track
        meta_track = midi.Track()
        time_sig = midi.TimeSignatureEvent()
        time_sig.set_numerator(4)
        time_sig.set_denominator(4)
        time_sig.set_metronome(metronome)
        time_sig.set_thirtyseconds(8)
        meta_track.append(time_sig)
        pattern.append(meta_track)

        # reshape to (SEQ_LENGTH X NUM_DIMS)
        sequence = np.reshape(sequence, [-1, self.note_range])

        time_steps = sequence.shape[0]
        if self.verbose:
            print "Total number of time steps: {}".format(time_steps)

        tick = time_step
        self.notes_on = { n: False for n in range(self.note_range) }
        # for seq_idx in range(188, 220):
        for seq_idx in range(time_steps):
            notes = np.nonzero(sequence[seq_idx, :])[0].tolist()

            # this tick will only be assigned to first NoteOn/NoteOff in
            # this time_step

            # NoteOffEvents come first so they'll have the tick value
            # go through all notes that are currently on and see if any
            # turned off
            for n in self.notes_on:
                if self.notes_on[n] and n not in notes:
                    tick = self.note_off(n, tick)
                    self.notes_on[n] = False

            # Turn on any notes that weren't previously on
            for note in notes:
                if not self.notes_on[note]:
                    tick = self.note_on(note, tick)
                    self.notes_on[note] = True

            tick += time_step

        # flush out notes
        for n in self.notes_on:
            if self.notes_on[n]:
                self.note_off(n, tick)
                tick = 0
                self.notes_on[n] = False

        pattern.append(self.track)
        midi.write_midifile(output_filename, pattern)

if __name__ == '__main__':
    pass
