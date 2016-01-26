import midi

# Default MIDI file resolution for datasets
RESOLUTION = 100 

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

    pattern = midi.Pattern(resolution=RESOLUTION)
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

if __name__ == '__main__':
    pass
