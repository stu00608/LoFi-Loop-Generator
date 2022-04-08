import pypianoroll
import numpy as np

def check_single_note(seqs):
    # This function is checking every sample from the measurement, to make sure
    # there is only one note on in one track, means the major sound and accompany
    # sound is separated.

    result = True
    for seq in seqs:
        for beat in seq:
            test = np.count_nonzero(beat!=0)
            if test > 1:
                result = False
                print(f"Error!, this case is {test}")
    
    return result

def beat_encode(beat):

    result = []
    for sample in beat:
        note_on_index = np.where(sample == 1)
        if(note_on_index[0].size > 0):
            # There is a note on.
            result.append(note_on_index[0][0])
        else:
            # There is no note on, a rest pattern.
            result.append(0)
        
    encoded_beat = []
    while(result):
        first_note = result[0]
        count = None
        for index, note in enumerate(result):
            if note != first_note:
                count = index
                break
            elif index == len(result)-1:
                count = len(result)
                break
        result = result[count:]
        encoded_beat.append(f"{first_note}-{count}")
    
    
    encoded_beat = '#'.join(encoded_beat)
    
    return encoded_beat

def beat_decode(encoded_beat):
    
    encoded_beat_list = encoded_beat.split('#')

    decoded_beat = []
    for note in encoded_beat_list:
        note_metadata = note.split('-')
        pitch = int(note_metadata[0])
        expanded_pitch = np.zeros(128)
        if(pitch != 0):
            # Not empty sample
            expanded_pitch[pitch] = 1
        count = int(note_metadata[1])
        decoded_beat += [expanded_pitch for _ in range(count)]
    
    decoded_beat = np.array(decoded_beat)

    return decoded_beat